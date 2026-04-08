"""
RAG 引擎模块（rag_engine.py）
============================
【职责】在已有 FAISS 向量库上完成：查询扩展 → 混合检索 →（可选）CrossEncoder 重排
        →（可选）LLM 上下文压缩 → 拼 Prompt → DeepSeek 生成 → 推理/最终答案拆分
        → concise 模式下的忠实度相关后处理。

【学习建议】先读 `RAGEngine.ask`（单次问答总入口），再读 `_answer_from_context_docs`、`_build_context`，
最后读本文件前半段的 `format_answer_with_reasoning` 与各 `clip_/expand_` 函数（concise 后处理）。
完整分支图见 docs/ARCHITECTURE.md「2.2 RAGEngine.ask 分支决策树」。

【数据流（与 ARCHITECTURE.md 第 4 节对应）】
  question ──► 离题短路（可选）→ _expand_query
       ──► 主路径（USE_RERANKER 且模型就绪）：_retrieve_for_rerank → _rerank_documents → _answer_from_context_docs
       或 压缩路径：compression_retriever.invoke → 再拼 Prompt + LLM（未开 rerank 时）
       或 回退：qa_chain.invoke 或 ensemble 取文档 + 手写 _llm_generate（尤其 stream 场景）
       ──► _build_context（父子块：优先 metadata.parent_content）
       ──► optimized_prompt.format(context, question)
       ──► _llm_generate（temperature=0；可选 stream_callback）
       ──► format_answer_with_reasoning → answer_only 供 RAGAS

【记忆】本模块不读取历史消息；多轮仅由 API/DB 存库，单轮 ask 只消费当前 question。

配置依赖：config 中 DeepSeek、SEARCH_K、RERANK_*、RAG_PROMPT_STYLE、RAG_LLM_MAX_TOKENS、get_current_prompt_template。

【源码分区】文件中用「# ========== … ==========」标题划分：导入 → 离题短路 → concise 后处理 →
检索辅助函数 → RAGEngine 类（__init__ / 检索 / LLM / ask）。
"""
import math  # 重排分数 NaN 时降级为 -inf，避免 sorted 行为不确定
import os
import re  # concise 模式下折叠换行，减少 RAGAS faithfulness 误拆句
from typing import Callable, List, Optional, Tuple

import config as _bootstrap_config  # noqa: F401 — 先于 langchain 加载 KMP/OMP 等，避免 Windows 下 native 崩溃

# ========== 依赖导入：多版本 LangChain 兼容（community / classic / 旧 langchain）==========
try:
    from langchain_community.chat_models import ChatOpenAI
    from langchain_community.retrievers import BM25Retriever
except ImportError:
    from langchain.chat_models import ChatOpenAI
    from langchain.retrievers import BM25Retriever

try:
    from langchain_classic.chains import RetrievalQA
    from langchain_classic.retrievers import EnsembleRetriever
    from langchain_classic.prompts import PromptTemplate
except ImportError:
    from langchain.chains import RetrievalQA
    from langchain.retrievers import EnsembleRetriever
    from langchain.prompts import PromptTemplate

# 上下文压缩相关导入 - 使用兼容性更好的导入方式
try:
    from langchain.retrievers.document_compressors import LLMChainExtractor
    from langchain.retrievers import ContextualCompressionRetriever
except ImportError:
    try:
        # 兼容旧版本 LangChain
        from langchain.retrievers.document_compressors import LLMChainExtractor
        from langchain.retrievers import ContextualCompressionRetriever
    except ImportError:
        print("[WARN] 上下文压缩功能不可用，将使用原始检索器")
        LLMChainExtractor = None
        ContextualCompressionRetriever = None

try:
    from langchain_core.messages import HumanMessage
    from langchain_core.documents import Document
except ImportError:
    from langchain.schema import HumanMessage
    from langchain.schema import Document

# 运行期常量：检索宽度、重排、Prompt 风格、离题词表等均来自 config，勿在 rag_engine 写死业务阈值
from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    RAG_LLM_MAX_TOKENS,
    RAG_OFF_TOPIC_ENABLED,
    RAG_OFF_TOPIC_KEYWORDS,
    RAG_OFF_TOPIC_REPLY,
    RAG_PROMPT_STYLE,
    RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN,
    RERANK_BATCH_SIZE,
    RERANK_MAX_PASSAGE_CHARS,
    RERANK_POOL_SIZE,
    RERANK_TOP_N,
    RERANK_DEVICE,
    RERANK_USE_FP16_CUDA,
    RERANKER_MODEL_PATH,
    SEARCH_K,
    USE_RERANKER,
    get_current_prompt_template,  # 配置化的 Prompt 管理
    RAG_SYSTEM_TYPE,             # 系统类型配置
)


# ========== 区块：离题短路（不调检索/LLM，降低无效成本）==========
# Prompt 模板在 config.py，通过 RAG_SYSTEM_TYPE / get_current_prompt_template 切换业务话术


def _query_matches_off_topic(query: str) -> bool:
    """子串命中 config.RAG_OFF_TOPIC_KEYWORDS 时视为明显非制度问题，走礼貌拒答短路。"""
    qn = (query or "").strip()
    if not qn:
        return False
    ql = qn.lower()
    for kw in RAG_OFF_TOPIC_KEYWORDS:
        k = (kw or "").strip()
        if not k:
            continue
        if k in qn:
            return True
        if k.lower() in ql:
            return True
    return False


def _polite_off_topic_payload() -> dict:
    """不调检索与 LLM，返回与 ask() 同结构的固定拒答（无引用）。"""
    reasoning = (
        "推理：该问题与员工考勤、薪酬、假期等制度无关，或属于外部常识/闲聊；"
        "本助手仅连接公司已上传的制度文档，无法回答此类内容。\n\n"
    )
    final_line = f"最终答案：{RAG_OFF_TOPIC_REPLY}"
    return {
        "result": f"{reasoning}{final_line}",
        "answer_only": RAG_OFF_TOPIC_REPLY,
        "source_documents": [],
    }


# ========== 区块：CrossEncoder 设备与 passage 截断（重排性能相关）==========


def _resolve_rerank_torch_device() -> str:
    """按 RERANK_DEVICE（auto/cuda/cpu）选择 CrossEncoder 所在设备，不可用时回退 cpu 并打印告警。"""
    import torch
    mode = (RERANK_DEVICE or "auto").strip().lower()
    if mode == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if mode == "cuda":
        if not torch.cuda.is_available():
            print("[WARN] RERANK_DEVICE=cuda 但 CUDA 不可用，回退 cpu（请安装 CUDA 版 PyTorch 与 NVIDIA 驱动）")
            return "cpu"
        return "cuda"
    return "cpu"


def _passage_text_for_cross_encoder(doc: Document) -> str:
    """
    供 CrossEncoder 打分的 passage：截断至 RERANK_MAX_PASSAGE_CHARS，避免长父块拖慢 tokenizer 与推理。
    子块为空时回退 metadata.parent_content 前缀，与重排语义仍大致一致。
    """
    meta = getattr(doc, "metadata", None) or {}
    t = (getattr(doc, "page_content", None) or "").strip()
    if not t and isinstance(meta, dict):
        t = (meta.get("parent_content") or "").strip()
    lim = RERANK_MAX_PASSAGE_CHARS
    if lim and len(t) > lim:
        return t[:lim]
    return t


# ========== 区块：concise 模式答案后处理（faithfulness / answer_relevancy 与 RAGAS 对齐）==========
# 思路：先规范化空白 → 按上下文剔除杜撰分句 → 多问问句数对齐 → 受控扩窗提高与问题的字面重合


def normalize_concise_rag_answer(text: str) -> str:
    """将 concise 风格下模型仍输出的换行折成单段空格，降低 faithfulness 多句与元格式噪声。"""
    if not text:
        return text
    flat = re.sub(r"[\r\n]+", " ", text.strip())
    return re.sub(r"[ \t　]+", " ", flat).strip()


def _merged_raw_doc_text(docs) -> str:
    """
    拼接当前检索文档的正文，供 concise 子句校验与扩窗（与 _build_context 对齐）。
    须包含 metadata.parent_content（若存在），否则仅 page_content 时答案常来自父块字面，
    导致「answer not in context_blob」、扩窗跳过，RAGAS answer_relevancy 易偏低。
    """
    parts = []
    for d in docs or []:
        meta = getattr(d, "metadata", None) or {}
        if isinstance(meta, dict):
            p = meta.get("parent_content")
            if isinstance(p, str) and p.strip():
                parts.append(p.strip())
        c = getattr(d, "page_content", "") or ""
        if c.strip():
            parts.append(c.strip())
    return "\n".join(parts)


def document_context_text_for_eval(doc) -> str:
    """与 _build_context 对齐：RAGAS contexts 优先父块，否则子块 page_content。"""
    meta = getattr(doc, "metadata", None) or {}
    if isinstance(meta, dict):
        p = meta.get("parent_content")
        if isinstance(p, str) and p.strip():
            return p.strip()
    return (getattr(doc, "page_content", None) or "").strip()


def clip_concise_clauses_by_context(answer: str, context_blob: str) -> str:
    """
    多「；」答案时：丢掉在检索正文中找不到连续字面（去空白后）匹配的分句，降低杜撰分句对 faithfulness 的拉低。
    单分句不做剔除，避免轻微增字导致误杀整答。
    """
    if not answer or not context_blob:
        return answer
    if "根据提供的文档，无法找到相关信息" in answer:
        return answer
    blob = re.sub(r"[\s\u3000]+", "", context_blob)
    raw_parts = [p.strip() for p in answer.split("；") if p.strip()]
    if len(raw_parts) <= 1:
        return answer
    kept = []
    for p in raw_parts:
        pc = re.sub(r"[\s\u3000]+", "", p)
        if len(pc) >= 2 and pc in blob:
            kept.append(p)
    if not kept:
        return answer
    return "；".join(kept)


def trim_redundant_clauses_for_multi_question(answer: str, question: str) -> str:
    """
    用户问题含多个「？」时，答案中「；」分句数常与问号数一致；多出的分句多为跑题摘录，砍掉可抬 answer_relevancy 且不伤 faithfulness（前提：保留段均为原文）。
    """
    if not answer or "根据提供的文档，无法找到相关信息" in answer:
        return answer
    qmarks = question.count("？") + question.count("?")
    if qmarks < 2:
        return answer
    parts = [p.strip() for p in answer.split("；") if p.strip()]
    if len(parts) <= qmarks:
        return answer
    return "；".join(parts[:qmarks])


def _question_char_overlap_score(text: str, question: str) -> int:
    """与用户问题在字符级的重合数（弱代理 relevance）。"""
    qset = set(re.sub(r"[？? \t，,。；、]", "", question))
    return sum(1 for ch in text if ch in qset)


def expand_single_clause_for_relevancy(answer: str, question: str, context_blob: str, margin: int) -> str:
    """
    单分句且为 context 子串时，向前后各扩 margin 字以内连续原文；
    仅当与用户问题的字符重合数严格增加时才采用，避免faithfulness 风险下无意义加长。
    """
    if margin <= 0 or not answer or not context_blob:
        return answer
    if "；" in answer:
        return answer
    if "根据提供的文档，无法找到相关信息" in answer:
        return answer
    if answer not in context_blob:
        return answer
    pos = context_blob.index(answer)
    start = max(0, pos - margin)
    end = min(len(context_blob), pos + len(answer) + margin)
    window = context_blob[start:end].strip()
    if _question_char_overlap_score(window, question) <= _question_char_overlap_score(answer, question):
        return answer
    if len(window) > len(answer) + 2 * margin + 8:
        return answer
    return window


def apply_concise_answer_postprocess(text: str, question: str, context_blob: str) -> str:
    """
    concise 流水线：空白归一 → `clip_concise_clauses_by_context`（剔除上下文无字面的「；」分句）
    → `trim_redundant_clauses_for_multi_question`（多问时对齐分句数）→ 可选 `expand_*` 扩窗。
    `context_blob` 须与检索正文一致（含 parent_content），见 `_merged_raw_doc_text`。
    """
    out = normalize_concise_rag_answer(text)
    out = clip_concise_clauses_by_context(out, context_blob)
    out = trim_redundant_clauses_for_multi_question(out, question)
    margin = RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN
    if margin <= 0:
        return out
    if "；" in out:
        parts = [p.strip() for p in out.split("；") if p.strip()]
        if not parts:
            return out
        expanded = [expand_single_clause_for_relevancy(p, question, context_blob, margin) for p in parts]
        return "；".join(expanded)
    return expand_single_clause_for_relevancy(out, question, context_blob, margin)


def split_reasoning_and_final_answer(text: str) -> Tuple[str, str]:
    """按行首「最终答案：」或「【最终答案】」拆成 (推理段, 最终段)；无标记则 ("", 全文)。"""
    if not text:
        return "", ""
    s = text.strip()
    m = re.search(r"(?m)^\s*最终答案[：:]\s*", s)
    if not m:
        m = re.search(r"(?m)^\s*【\s*最终答案\s*】\s*", s)
    if m:
        return s[: m.start()].strip(), s[m.end() :].strip()
    return "", s


def merge_reasoning_display(reasoning: str, final_processed: str) -> str:
    """拼接展示用全文：保留推理段 + 标准化后的最终答案标记。"""
    if reasoning:
        return f"{reasoning}\n\n最终答案：{final_processed}"
    return final_processed


def _extract_stream_chunk_text(chunk) -> str:
    """从 ChatModel 流式 chunk 中取出增量文本（兼容 str / content 块列表）。"""
    if chunk is None:
        return ""
    c = getattr(chunk, "content", None)
    if c is None:
        return ""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for block in c:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "") or ""))
            else:
                parts.append(str(getattr(block, "text", block) or ""))
        return "".join(parts)
    return str(c)


# ========== 区块：LLM 原始输出 → 展示用 result + 评测用 answer_only ==========


def format_answer_with_reasoning(
    raw_llm_text: str, query: str, context_blob: str, concise: bool
) -> Tuple[str, str]:
    """
    解析模型输出：先 `split_reasoning_and_final_answer` 拆「推理」与「最终答案」；
    concise 时只对最终段做 `apply_concise_answer_postprocess`，再 `merge_reasoning_display` 拼回展示形态。
    返回 (result 用全文, answer_only 仅最终段)；RAGAS 等评测消费第二项，避免推理前缀干扰指标。
    """
    reasoning, final = split_reasoning_and_final_answer(raw_llm_text)
    body = (final or raw_llm_text or "").strip()
    if concise:
        body = apply_concise_answer_postprocess(body, query, context_blob)
    display = merge_reasoning_display(reasoning, body)
    return display, body


# ========== 区块：检索前处理（去重、BM25 文档源、路径检查）==========


def _dedupe_documents(docs) -> List:
    """按正文前缀去重，保持顺序（先向量后 BM25）。"""
    seen = set()
    out = []
    for d in docs:
        key = hash(d.page_content[:800])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def _reranker_path_ok(path: str) -> bool:
    """检查本地 reranker 目录是否像 HuggingFace 快照（含 config.json）。"""
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _faiss_documents_for_bm25(vectorstore, k_max: int = 1000) -> List:
    """
    从 FAISS 按索引顺序遍历 docstore，取出至多 k_max 条文档供 BM25 建库。
    避免使用 similarity_search('')：空串嵌入在部分环境下会触发异常或底层 native 崩溃。
    """
    try:
        n = int(vectorstore.index.ntotal)
    except Exception:
        n = 0
    if n <= 0:
        return []
    out: List = []
    take = min(n, k_max)
    for i in range(take):
        try:
            doc_id = vectorstore.index_to_docstore_id[i]
            doc = vectorstore.docstore.search(doc_id)
            if isinstance(doc, Document):
                out.append(doc)
        except Exception:
            continue
    if out:
        return out
    return vectorstore.similarity_search("。", k=min(k_max, max(n, 1)))


# ========== 类 RAGEngine：向量+BM25 混合检索、可选重排、Prompt+LLM、流式 ==========


class RAGEngine:
    """
    混合检索 + 可选 CrossEncoder 重排 + DeepSeek 生成。
    初始化阶段：建 retriever / ensemble /（可选）qa_chain 与 compression_retriever；
    运行阶段：入口一律为 ask()。
    """

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore  # LangChain FAISS 包装，含 embed_query 与 docstore
        # LLM：temperature=0 降随机性；max_tokens 须顶层传入（LangChain Pydantic 禁止塞 model_kwargs）
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            n=1,
            max_tokens=int(RAG_LLM_MAX_TOKENS),
        )

        # 限制 BLAS 线程，降低与 PyTorch OpenMP 在 BM25/NumPy 路径上并发导致的 Windows 段错误概率
        # 注意：新版 numpy (2.x) 已移除 set_num_threads 方法，按官方建议使用环境变量控制
        # 环境变量已在 .env 中设置，这里仅做兼容处理
        try:
            import numpy as _np
            print(f"[INFO] numpy 版本: {_np.__version__} （已通过 .env 环境变量控制线程）")
        except Exception as e:
            print(f"[WARN] numpy 初始化异常: {e}")

        # 【重要】在创建检索器前再次强化环境变量，防止段错误
        import os
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        print("[DEBUG] RAGEngine 中再次强制设置单线程环境变量")

        # 向量一路：每问一次 embed_query + FAISS 近邻
        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": SEARCH_K}
        )
        # BM25 一路：依赖预先扫 docstore 建稀疏索引（与向量互补，英文/专有名词有时更稳）
        try:
            docs = _faiss_documents_for_bm25(self.vectorstore, k_max=1000)
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = SEARCH_K
            print("[OK] BM25Retriever 创建成功")
        except Exception as bm25_err:
            print(f"[ERROR] BM25Retriever 创建失败: {bm25_err}")
            raise

        # 加权合并两路结果（权重和不必为 1，LangChain 内部会归一）
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.4, 0.6],
        )

        # Prompt：仅两占位符 context / question，与下面手写 format 一致
        prompt_template_text = get_current_prompt_template()
        
        self.optimized_prompt = PromptTemplate(
            template=prompt_template_text,
            input_variables=["context", "question"],
        )
        print(f"[INFO] RAG 生成模版: 系统类型={RAG_SYSTEM_TYPE}, Prompt样式={RAG_PROMPT_STYLE}")

        # 重排就绪则主路径用手写检索+精排，不再用 RetrievalQA 包一层（避免与 rerank 逻辑重复）
        self._cross_encoder = None
        self._reranker_ready = bool(USE_RERANKER and _reranker_path_ok(RERANKER_MODEL_PATH))

        if self._reranker_ready:
            self.qa_chain = None
        else:
            # 无重排时：经典 stuff 链，一次 invoke 完成检索+生成
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.ensemble_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.optimized_prompt},
            )

        # 可选第二路：对 ensemble 结果再用 LLM 做摘录（成本高；开 rerank 时 ask 不优先走此路）
        self.compression_retriever = None
        if LLMChainExtractor is not None and ContextualCompressionRetriever is not None:
            try:
                compressor = LLMChainExtractor.from_llm(self.llm)
                self.compression_retriever = ContextualCompressionRetriever(
                    base_compressor=compressor,
                    base_retriever=self.ensemble_retriever
                )
                print("[INFO] 上下文压缩检索器已启用")
            except Exception as e:
                print(f"[WARN] 上下文压缩器初始化失败: {e}")
                self.compression_retriever = None
        else:
            print("[INFO] 上下文压缩功能不可用，将使用原始检索器")

    # ---------- RAGEngine / CrossEncoder：懒加载 + 失败则回退 qa_chain ----------
    def _ensure_cross_encoder(self):
        """
        懒加载 CrossEncoder（sentence-transformers），供 `_rerank_documents` 与 API 启动预热调用。
        - 成功：`_cross_encoder` 非空，精排走 GPU/CPU 前向。
        - 失败：置 `_cross_encoder_load_failed`，并补建 `qa_chain` 与无 rerank 时一致，避免后续 ask 崩。
        """
        if not self._reranker_ready:
            print("[DEBUG] reranker 未启用，跳过加载")
            return
        if self._cross_encoder is not None:
            return
        if getattr(self, "_cross_encoder_load_failed", False):
            print("[DEBUG] reranker 之前已加载失败，跳过")
            return
        try:
            print(f"[DEBUG] 开始加载 CrossEncoder: {RERANKER_MODEL_PATH}")
            import torch
            from sentence_transformers import CrossEncoder

            device = _resolve_rerank_torch_device()
            print(f"[INFO] CrossEncoder 使用设备: {device}")
            model_kwargs = {}
            if device == "cuda" and RERANK_USE_FP16_CUDA:
                model_kwargs["torch_dtype"] = torch.float16
            if model_kwargs:
                try:
                    self._cross_encoder = CrossEncoder(
                        RERANKER_MODEL_PATH, device=device, model_kwargs=model_kwargs
                    )
                except Exception as fp16_err:
                    print(f"[WARN] CrossEncoder FP16 加载失败，回退 FP32: {fp16_err}")
                    self._cross_encoder = CrossEncoder(RERANKER_MODEL_PATH, device=device)
            else:
                self._cross_encoder = CrossEncoder(RERANKER_MODEL_PATH, device=device)
            print(f"[INFO] CrossEncoder reranker loaded: {RERANKER_MODEL_PATH} ({device})")
        except Exception as e:
            self._cross_encoder_load_failed = True
            self._cross_encoder = None
            print(f"[ERROR] CrossEncoder 加载失败: {e}")
            print("[WARN] 因 reranker 加载失败，切换到 ensemble RetrieverQA")
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.ensemble_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.optimized_prompt},
            )

    # ---------- RAGEngine / 查询改写：仅关键词表驱动，减轻同义词漏召 ----------
    def _expand_query(self, query: str) -> str:
        """
        若问题命中预设 HR 词（试用期、年假等），把同义词拼进查询串，供向量与 BM25 共用。
        不改变用户原始 `query` 在 Prompt【问题】中的展示（精排仍用原始 query 与 passage 配对）。
        """
        expansions = {
            "试用期": ["试用期", "转正", "试用", " probation period"],
            "年假": ["年假", "年休假", "带薪年假", "annual leave"],
            "离职": ["离职", "辞职", "离开公司", "resignation"],
            "工资": ["工资", "薪资", "薪酬", "salary"],
            "请假": ["请假", "休假", "病假", "事假", "leave"],
        }
        for keyword, synonyms in expansions.items():
            if keyword in query:
                return f"{query} {' '.join(synonyms)}"
        return query

    def _bm25_invoke(self, q: str):
        """统一 BM25 调用：LangChain 新版 `invoke` 与旧版 `get_relevant_documents` 二选一。"""
        r = self.bm25_retriever
        if hasattr(r, "invoke"):
            return r.invoke(q)
        return r.get_relevant_documents(q)

    # ---------- RAGEngine / 重排前候选池：向量 + BM25 合并，限制条数防 CE 过慢 ----------
    def _retrieve_for_rerank(self, expanded_query: str) -> List:
        """
        双路各取至多 `RERANK_POOL_SIZE` 条，再 `_dedupe_documents` 合并去重。
        使用 `expanded_query` 扩大召回；BM25 的 `k` 在此临时调小/对齐，用完 `finally` 恢复，避免污染全局 retriever。
        """
        kv = max(1, int(RERANK_POOL_SIZE))
        vdocs = self.vectorstore.similarity_search(expanded_query, k=kv)
        r = self.bm25_retriever
        prev_k = getattr(r, "k", SEARCH_K)
        try:
            r.k = min(kv, prev_k) if prev_k else kv
            bdocs = self._bm25_invoke(expanded_query)
        finally:
            r.k = prev_k
        return _dedupe_documents(list(vdocs) + list(bdocs))

    def _rerank_documents(self, query: str, docs: List) -> List:
        """
        用 **原始问题** `query` 与每条 passage（经 `_passage_text_for_cross_encoder` 截断）组成 pair，批量 `predict`。
        分数降序取前 `RERANK_TOP_N`；异常或模型未加载时退回「原始顺序截断」，保证 ask 总能返回。
        """
        self._ensure_cross_encoder()
        if not docs or self._cross_encoder is None:
            print("[DEBUG] CrossEncoder 未加载，使用原始文档顺序")
            return docs[:RERANK_TOP_N] if docs else []
        
        print(f"[DEBUG] 开始 rerank {len(docs)} 个文档")
        pairs = [[query, _passage_text_for_cross_encoder(d) or (d.page_content or "")[:200]] for d in docs]
        
        try:
            scores = self._cross_encoder.predict(
                pairs,
                batch_size=max(1, RERANK_BATCH_SIZE),
                show_progress_bar=False,
            )
            print(f"[DEBUG] CrossEncoder predict 成功，返回 {len(scores)} 个分数")
        except Exception as e:
            print(f"[ERROR] CrossEncoder predict 失败: {e}")
            print("[WARN] rerank 失败，回退到原始顺序")
            return docs[:RERANK_TOP_N] if docs else []

        def _rerank_score_key(s):
            """将 CrossEncoder 输出转为可排序的标量；NaN/不当无穷大视为最低分。"""
            x = float(s)
            if math.isnan(x):
                return float("-inf")
            if math.isinf(x) and x < 0:
                return float("-inf")
            return x

        scored = sorted(zip(scores, docs), key=lambda t: _rerank_score_key(t[0]), reverse=True)
        return [d for _, d in scored[:RERANK_TOP_N]]

    # ---------- RAGEngine / 拼 Prompt 上下文：与 knowledge_base 父子块约定一致 ----------
    def _build_context(self, docs: List) -> str:
        """
        将 Top 文档拼成一大段 `context` 填入模板【上下文】。
        子块 metadata 若含 `parent_content`（建库时由 KnowledgeBase 写入），则用父块全文，否则用子块 `page_content`。
        这样检索命中细粒度子块，生成仍能看到更完整段落，减轻断章取义。
        """
        parts = []
        for i, d in enumerate(docs, 1):
            # 优先使用父文档内容（语义更完整）
            if "parent_content" in d.metadata and d.metadata.get("parent_content"):
                content = d.metadata["parent_content"]
                source_type = "父文档"
            else:
                content = d.page_content
                source_type = "子文档"
            parts.append(f"[片段{i}]({source_type})\n{content}")
        return "\n\n".join(parts)

    def _ensemble_get_documents(self, expanded_query: str) -> List:
        """
        仅拉取 ensemble 文档列表，不调用 LLM；用于流式场景（RetrievalQA 无法边生成边回调 token）。
        与 `qa_chain` 底层 retriever 一致，保证非流式与流式检索结果同源。
        """
        er = self.ensemble_retriever
        if hasattr(er, "invoke"):
            return list(er.invoke(expanded_query) or [])
        return list(er.get_relevant_documents(expanded_query) or [])

    # ---------- RAGEngine / LLM：非流式 invoke 与流式 stream 合一出口 ----------
    def _llm_generate(
        self, prompt_text: str, stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        组装单条 `HumanMessage` 调用 DeepSeek（经 LangChain ChatOpenAI）。
        - 无 `stream_callback`：同步 `invoke`，适合 `/api/chat` 线程池路径。
        - 有 `stream_callback`：逐 chunk 解析文本并回调，最后拼接成完整串，供后处理与落库。
        """
        messages = [HumanMessage(content=prompt_text)]
        if stream_callback is None:
            resp = self.llm.invoke(messages)
            return resp.content if hasattr(resp, "content") else str(resp)
        acc: List[str] = []
        try:
            stream_iter = self.llm.stream(messages)
        except Exception as e:
            print(f"[WARN] LLM stream 失败，回退 invoke: {e}")
            resp = self.llm.invoke(messages)
            return resp.content if hasattr(resp, "content") else str(resp)
        for chunk in stream_iter:
            piece = _extract_stream_chunk_text(chunk)
            if piece:
                acc.append(piece)
                stream_callback(piece)
        return "".join(acc)

    def _answer_from_context_docs(
        self,
        query: str,
        docs: List,
        stream_callback: Optional[Callable[[str], None]] = None,
    ) -> dict:
        """
        **重排路径与兜底路径的汇合点**：已定稿 `docs` → `_build_context` → `optimized_prompt.format` → `_llm_generate`
        → `format_answer_with_reasoning`（concise 后处理、拆 `answer_only`）。
        返回 dict 与 `ask` 其它分支一致，便于 API 与评测统一解析。
        """
        context = self._build_context(docs)
        prompt_text = self.optimized_prompt.format(context=context, question=query)
        raw = self._llm_generate(prompt_text, stream_callback)
        blob = _merged_raw_doc_text(docs)
        use_concise = RAG_PROMPT_STYLE == "concise"
        display, answer_only = format_answer_with_reasoning(raw, query, blob, use_concise)
        return {"result": display, "answer_only": answer_only, "source_documents": docs}

    def ask(self, query, stream_callback: Optional[Callable[[str], None]] = None):
        """
        单次问答入口（无多轮历史注入）。
        :param query: 当前用户问题（整段进入 Prompt 的【问题】栏；CrossEncoder 也用它与 passage 配对）。
        :param stream_callback: 非 None 时 LLM 使用 stream，并对每个文本片回调（SSE 用）。
        :return: dict：`result`（界面展示，可含推理前缀）、`answer_only`（最终句，供 RAGAS）、`source_documents`（引用列表）。

        分支优先级简述：离题短路 →（若启用 rerank）双路召回+CE→`_answer_from_context_docs`；
        否则若有压缩检索器则走压缩链；否则 `qa_chain` 或 ensemble 手写 LLM；最后防御性兜底。
        """
        # 以下分支顺序与 docs/ARCHITECTURE.md「2.2 决策树」一致，修改时请同步文档
        print(f"[DEBUG] === ask() 开始执行，query: {query[:50]}... ===")

        if RAG_OFF_TOPIC_ENABLED and _query_matches_off_topic(query):
            print("[INFO] 命中离题词表，返回礼貌拒答（不调检索/LLM）")
            payload = _polite_off_topic_payload()
            if stream_callback:
                stream_callback(payload["result"])
            return payload

        # 步骤 1：轻量查询扩展，提高 BM25/向量对同义词的召回
        expanded_query = self._expand_query(query)
        print(f"[DEBUG] query 扩展完成: {expanded_query[:50]}...")

        # 步骤 2：启用 Rerank 时优先 CrossEncoder（不先走压缩检索器，避免每条候选一次 LLM、约 10s+ 且精排不生效）
        if self._reranker_ready:
            print("[DEBUG] reranker 已启用，走 CrossEncoder 精排链路")
            self._ensure_cross_encoder()
            merged = self._retrieve_for_rerank(expanded_query)
            top_docs = self._rerank_documents(query, merged)
            return self._answer_from_context_docs(query, top_docs, stream_callback)

        print("[DEBUG] reranker 未启用")

        # 步骤 3a：无 rerank 时，若初始化了 compression_retriever，则先让 LLM 从 ensemble 结果里「摘录」再生成第二道 LLM
        # （两次 LLM，成本高；与步骤 2 的 CE 精排互斥为主路径）
        if hasattr(self, 'compression_retriever') and self.compression_retriever is not None:
            print("[DEBUG] 使用上下文压缩检索器")
            compressed_docs = self.compression_retriever.invoke(expanded_query)
            if RAG_PROMPT_STYLE == "concise":
                context = self._build_context(compressed_docs)
                prompt_text = self.optimized_prompt.format(context=context, question=query)
                raw = self._llm_generate(prompt_text, stream_callback)
                blob = _merged_raw_doc_text(compressed_docs)
                display, answer_only = format_answer_with_reasoning(raw, query, blob, True)
                return {
                    "result": display,
                    "answer_only": answer_only,
                    "source_documents": compressed_docs,
                }
            else:
                # standard：非流式可走 qa_chain；流式必须手写「ensemble 取 docs + format + _llm_generate」
                if stream_callback is not None:
                    docs = self._ensemble_get_documents(expanded_query)
                    ctx_docs = docs[:RERANK_TOP_N] if docs else []
                    # context 用全部 ensemble 文档；source_documents 只截 RERANK_TOP_N 与列表展示一致
                    context = self._build_context(docs)
                    prompt_text = self.optimized_prompt.format(context=context, question=query)
                    raw = self._llm_generate(prompt_text, stream_callback)
                    blob = _merged_raw_doc_text(ctx_docs)
                    display, answer_only = format_answer_with_reasoning(raw, query, blob, False)
                    return {
                        "result": display,
                        "answer_only": answer_only,
                        "source_documents": ctx_docs,
                    }
                result = self.qa_chain.invoke({"query": expanded_query})
                if result.get("source_documents"):
                    result["source_documents"] = result["source_documents"][:RERANK_TOP_N]
                if result.get("result") is not None:
                    blob = _merged_raw_doc_text(result.get("source_documents") or [])
                    display, answer_only = format_answer_with_reasoning(
                        str(result["result"]), query, blob, False
                    )
                    result["result"] = display
                    result["answer_only"] = answer_only
                return result

        # 步骤 3b：无压缩器（或未进入 3a）且已有 qa_chain：非流式一次 invoke；流式同样手写 ensemble+LLM
        if self.qa_chain is not None:
            if stream_callback is not None:
                docs = self._ensemble_get_documents(expanded_query)
                ctx_docs = docs[:RERANK_TOP_N] if docs else []
                context = self._build_context(docs)
                prompt_text = self.optimized_prompt.format(context=context, question=query)
                raw = self._llm_generate(prompt_text, stream_callback)
                blob = _merged_raw_doc_text(ctx_docs)
                use_concise = RAG_PROMPT_STYLE == "concise"
                display, answer_only = format_answer_with_reasoning(raw, query, blob, use_concise)
                return {
                    "result": display,
                    "answer_only": answer_only,
                    "source_documents": ctx_docs,
                }
            result = self.qa_chain.invoke({"query": expanded_query})
            if result.get("source_documents"):
                result["source_documents"] = result["source_documents"][:RERANK_TOP_N]
            if result.get("result") is not None:
                docs_ctx = result.get("source_documents") or []
                blob = _merged_raw_doc_text(docs_ctx)
                use_concise = RAG_PROMPT_STYLE == "concise"
                display, answer_only = format_answer_with_reasoning(
                    str(result["result"]), query, blob, use_concise
                )
                result["result"] = display
                result["answer_only"] = answer_only
            return result

        # 步骤 3c：理论上不应到达（未开 Rerank 时 __init__ 已创建 qa_chain）；防御性退回 ensemble 截断
        print("[WARN] RAG 路径异常：qa_chain 为空，退回 ensemble 检索截断")
        docs = self._ensemble_get_documents(expanded_query)
        ctx_docs = docs[:RERANK_TOP_N] if docs else []
        return self._answer_from_context_docs(query, ctx_docs, stream_callback)
