"""
HR 制度 RAG 问答系统 — 命令行入口
================================
运行：python main.py
流程：加载/构建 FAISS → 构造 RAGEngine → 循环读取用户输入 → 流式打印回答。
说明：CLI 不落库 MySQL；多轮对话在模型侧仍按「单轮」处理（无历史注入），与 ARCHITECTURE 4.5 一致。

【与 api_server 的差异】二者都构造 `KnowledgeBase` + `RAGEngine`，但 CLI 无 FastAPI lifespan、无 `to_thread`、无 MySQL。
学习时可先跑通本文件再对照 `api_server._bootstrap_rag_core`。
"""
import os  # 在 import torch/langchain 之前设置线程环境变量
import sys  # 标准流重绑定为 UTF-8（Windows 控制台）

# 须最先执行：减轻 Windows 下 OpenMP / MKL 与 PyTorch 多库并发导致的段错误
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 将 stdout 包装为 UTF-8，避免中文乱码
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# 项目内模块（config 会加载 .env）
from config import FAISS_INDEX_PATH, get_seed_document_path
from knowledge_base import KnowledgeBase
from rag_engine import RAGEngine


def main():
    """
    交互式主循环：构建或加载向量库 → `RAGEngine` → 每轮 `ask`。
    与 API 差异：无 `to_thread`（主线程直接跑 RAG）、无 MySQL；流式仅 stdout 回调。
    """
    print("=" * 60)
    print("HR制度RAG问答系统")
    print("=" * 60)

    # 嵌入模型与 FAISS 封装
    kb = KnowledgeBase()

    # 无本地索引目录则首次建库：按 get_seed_document_path() 解析种子文件并切块、写入 FAISS
    if not os.path.exists(FAISS_INDEX_PATH):
        print("[Start] 首次运行，正在构建知识库...")
        file_path = get_seed_document_path()
        print(f"[Start] 使用种子文档: {file_path}")
        texts = kb.process_document(file_path)
        vectorstore = kb.create_vector_store(texts)
    else:
        # 已有索引则直接加载，避免重复嵌入计算
        vectorstore = kb.load_vector_store()

    # 混合检索 + LLM 生成引擎
    engine = RAGEngine(vectorstore)

    print("\n[OK] 系统已就绪！请输入问题 (输入 'exit' 退出):")

    while True:
        query = input("\n[User] ")
        if query.lower() == "exit":
            break

        print("[AI] ", end="", flush=True)
        # 流式：仅 LLM 生成阶段回调；前面的检索/重排仍阻塞至首 token 前
        result = engine.ask(
            query,
            stream_callback=lambda t: (sys.stdout.write(t), sys.stdout.flush()),
        )
        print()

        # 简要展示引用片段（前 3 条），便于对照制度原文
        if result.get("source_documents"):
            print("[Doc] 参考来源:")
            for i, doc in enumerate(result["source_documents"][:3]):
                preview = doc.page_content[:80].replace("\n", " ")
                print(f"  [{i+1}] {preview}...")


if __name__ == "__main__":
    main()
