"""
知识库模块（knowledge_base.py）
==============================
【职责】
  - 加载 HuggingFace 中文嵌入（本地优先，否则 Hub）
  - 读取 PDF / DOCX / TXT，清洗文本
  - **父子切块**（见 ARCHITECTURE 4.3）：子块写入 FAISS，父正文挂在子块 metadata
  - FAISS 的 save_local / load_local

【与 rag_engine 的衔接】
  检索命中的是「子块」；生成上下文时 `_build_context` 读取 `parent_content` 扩大可见原文。

配置依赖：config.EMBEDDING_*、LOCAL_MODEL_PATH、HF_ENDPOINT、FAISS_INDEX_PATH。

【数据流】磁盘文件 → Loader → 清洗 → 父切分 → 子切分（子带 parent_content）→ FAISS 仅存子块；
推理时 rag_engine._build_context 优先用 parent_content 拼进 LLM。
"""
import os
import re

import config as _bootstrap_config  # noqa: F401 — 先于 langchain 应用 OpenMP/tokenizers 环境，减轻 Windows 段错误

# LangChain 1.x: 组件分散在多个包中
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # LangChain 0.2.2+ 推荐
    from langchain_community.document_loaders import TextLoader, PyPDFLoader
except ImportError:
    # 旧版兼容 (LangChain 0.1.x)
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.document_loaders import TextLoader, PyPDFLoader

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # LangChain 1.x
except ImportError:
    # 旧版兼容
    from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

from config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_DEVICE,
    HF_ENDPOINT,
    LOCAL_MODEL_PATH,
    FAISS_INDEX_PATH,
)


def _resolve_embedding_device():
    """
    根据 EMBEDDING_DEVICE 配置选择推理设备。
    "auto" 时检测 CUDA 可用性；否则按配置字符串（cuda/cpu）。
    在设备解析前再次强化线程设置，减少后续段错误概率。
    """
    # 在任何 torch 导入前再次设置环境变量
    import os
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    print("[DEBUG] knowledge_base 中预先设置线程环境变量")

    if EMBEDDING_DEVICE == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    if EMBEDDING_DEVICE == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            print("[WARN] EMBEDDING_DEVICE=cuda 但 torch.cuda.is_available()=False，回退 cpu（请安装 CUDA 版 PyTorch 与驱动）")
            return "cpu"
        except ImportError:
            print("[WARN] 未安装 torch，嵌入使用 cpu")
            return "cpu"
    return EMBEDDING_DEVICE


def _check_local_model_available(local_path: str) -> bool:
    """
    检查本地模型路径是否存在且包含必要的模型文件（如 config.json、pytorch_model.bin 等）。
    """
    if not os.path.isdir(local_path):
        return False
    required_files = ["config.json"]
    return all(os.path.isfile(os.path.join(local_path, f)) for f in required_files)


def _clean_text(raw: str) -> str:
    """
    文档级清洗：统一换行/空白/零宽字符等，减少无意义 token。
    """
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[\u200b\ufeff\u3000]", "", text)
    return text.strip()


def _clean_documents(documents):
    """
    对 LangChain Document 列表逐片清洗。
    """
    for doc in documents:
        doc.page_content = _clean_text(doc.page_content)
    return documents


class KnowledgeBase:
    """
    嵌入模型 + 文档处理 + FAISS 生命周期。

    【主要方法】
      - `process_document(path)`：读 PDF/DOCX/TXT → 父子切块 → 返回 Document 列表（尚未写入向量库）。
      - `create_vector_store(documents)`：嵌入并 `save_local` 到 `FAISS_INDEX_PATH`。
      - `load_vector_store()`：从磁盘恢复 FAISS + 同一套 Embeddings，供 `RAGEngine` 使用。

    典型用法：首次 `process_document` → `create_vector_store`；之后启动只 `load_vector_store`。
    """

    # ---------- 初始化：解析本地模型路径 + 设备 + HuggingFaceEmbeddings ----------
    def __init__(self):
        # 国内镜像：在创建 HuggingFaceEmbeddings 之前写入 HF_ENDPOINT，以下载/解析模型 ID
        if HF_ENDPOINT:
            os.environ["HF_ENDPOINT"] = HF_ENDPOINT
            print(f"[HF] 使用镜像 endpoint: {HF_ENDPOINT}")

        # 嵌入模型路径：本地目录完整则用本地，避免重复下载；否则回退到 Hub 上的 EMBEDDING_MODEL_NAME
        if _check_local_model_available(LOCAL_MODEL_PATH):
            model_path = LOCAL_MODEL_PATH
            print(f"[INFO] 使用本地模型: {model_path}")
        else:
            model_path = EMBEDDING_MODEL_NAME
            if LOCAL_MODEL_PATH:
                print(f"[WARNING] 配置的本地模型路径不存在或不完整: {LOCAL_MODEL_PATH}")
                print(f"[HINT] 运行 'python download_model.py' 可一键下载模型到本地")
            print(f"[INFO] 将从 HuggingFace Hub 在线加载: {model_path}")

        device = _resolve_embedding_device()
        print(f"[INFO] 正在加载 Embedding 模型（设备: {device}）...")

        # normalize_embeddings=True：余弦空间更稳定，与 FAISS 常用配置一致
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_path,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("[INFO] Embedding 模型加载完成")

    # ---------- 建库第一步：单文件 → 子块 Document 列表（尚未写入向量）----------
    def process_document(self, file_path):
        """
        加载单文件并切分为 **子块** Document 列表（每个子块 metadata 含父全文）。
        支持：.pdf（PyPDFLoader）、.docx（Docx2txtLoader）、其他扩展名按 UTF-8 文本（TextLoader）。
        返回的列表直接交给 `create_vector_store` 做嵌入与 FAISS 写入。
        """
        print(f"[INFO] 正在处理文档: {file_path}")

        # 按扩展名选择 Loader
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".docx"):
            try:
                from langchain_community.document_loaders import Docx2txtLoader
                loader = Docx2txtLoader(file_path)
            except ImportError:
                print("[ERROR] 缺少 docx 支持，请运行: pip install python-docx")
                raise
        else:
            loader = TextLoader(file_path, encoding="utf-8")

        documents = loader.load()

        # 清洗
        documents = _clean_documents(documents)

        # ---------- 父子索引（Parent-Child）----------
        # 子块：向量检索粒度细，易命中；父块：拼进 LLM 时语义更完整，减少断句幻觉。
        # 1）父文档切分（偏大）
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1200,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "！", "？"],
            length_function=len,
        )
        
        # 2）子文档切分（偏小，实际入索引）
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=30,
            separators=["\n\n", "\n", "。", "！", "？", "；", "，"],
            length_function=len,
        )
        
        parent_docs = parent_splitter.split_documents(documents)
        abs_source = os.path.abspath(file_path)

        # 每个父块再拆成若干子块；子块携带父全文供 rag_engine._build_context 使用
        child_docs = []
        for i, parent in enumerate(parent_docs):
            children = child_splitter.split_documents([parent])
            for child in children:
                child.metadata["parent_id"] = i  # 父序号，便于溯源
                child.metadata["parent_content"] = parent.page_content  # 生成用宽上下文
                child.metadata["is_parent"] = False
                child.metadata.setdefault("source", abs_source)
                child_docs.append(child)

        print(f"[OK] 父子索引策略：{len(parent_docs)} 个父文档，{len(child_docs)} 个子文档")

        return child_docs

    # ---------- 建库第二步：嵌入 + 落盘（与 load_vector_store 成对）----------
    def create_vector_store(self, texts):
        """
        用当前 `self.embeddings` 将文档块写入内存 FAISS，再 `save_local` 到 `FAISS_INDEX_PATH`。
        `texts` 一般为 `process_document` 返回的子块列表。
        """
        print("[INFO] 正在构建向量索引...")
        vectorstore = FAISS.from_documents(texts, self.embeddings)

        vectorstore.save_local(FAISS_INDEX_PATH)
        print(f"[OK] 索引已保存至: {FAISS_INDEX_PATH}")
        return vectorstore

    def load_vector_store(self):
        """
        从 `FAISS_INDEX_PATH` 加载已保存索引；须与建库时使用**同一套** Embeddings 模型与维度。
        `allow_dangerous_deserialization=True` 为 LangChain 加载本地 pickle 类索引所需（自有索引可信）。
        """
        if os.path.exists(FAISS_INDEX_PATH):
            print("[INFO] 正在加载本地索引...")
            return FAISS.load_local(
                FAISS_INDEX_PATH,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
        print("[WARN] 未找到本地索引，请先构建。")
        return None
