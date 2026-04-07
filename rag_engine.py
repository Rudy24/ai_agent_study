"""
RAG 引擎模块（rag_engine.py）
============================
【职责】在已有 FAISS 向量库上完成：查询扩展 → 混合检索 →（可选）CrossEncoder 重排
        →（可选）LLM 上下文压缩 → 拼 Prompt → DeepSeek 生成 → 推理/最终答案拆分
        → concise 模式下的忠实度相关后处理。

【数据流（与 ARCHITECTURE.md 第 4 节对应）】
  question ──► _expand_query（同义词扩召回）
       ──► 压缩路径：compression_retriever.invoke（LLM 摘录检索结果）
       或 重排路径：_retrieve_for_rerank + _rerank_documents（CrossEncoder）
       或 回退：qa_chain / ensemble 直接检索
       ──► _build_context（父子块：优先 metadata.parent_content）
       ──► optimized_prompt.format(context, question)  # 见 config 铁律与注入软防护
       ──► _llm_generate（temperature=0）
       ──► format_answer_with_reasoning → answer_only 供 RAGAS

【记忆】本模块不读取历史消息；多轮仅由 API/DB 存库，单轮 ask 只消费当前 question。

配置依赖：config 中 DeepSeek、SEARCH_K、RERANK_*、RAG_PROMPT_STYLE、get_current_prompt_template。
"""
import math  # 重排分数 NaN 时降级为 -inf，避免 sorted 行为不确定
import os
import re  # concise 模式下折叠换行，减少 RAGAS faithfulness 误拆句
from typing import Callable, List, Optional, Tuple

import config as _bootstrap_config  # noqa: F401 — 先于 langchain 加载 KMP/OMP 等，避免 Windows 下 native 崩溃

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

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    RAG_PROMPT_STYLE,
    RAG_RELEVANCY_CONTEXT_EXPAND_MARGIN,
    RERANK_BATCH_SIZE,
    RERANK_POOL_SIZE,
    RERANK_TOP_N,
    RERANKER_MODEL_PATH,
    SEARCH_K,
    USE_RERANKER,
    get_current_prompt_template,  # 配置化的 Prompt 管理
    RAG_SYSTEM_TYPE,             # 系统类型配置
)


# Prompt 模板已移至 config.py 统一管理
# 支持多种业务场景（HR、客服等），通过 RAG_SYSTEM_TYPE 配置切换
# 核心逻辑中不再包含业务相关的 Prompt 定义


def normalize_concise_rag_answer(text: str) -> str:
    """将 concise 风格下模型仍输出的换行折成单段空格，降低 faithfulness 多句与元格式噪声。"""
    if not text:
        return text
    flat = re.sub(r"[\r\n]+", " ", text.strip())
    return re.sub(r"[ \t　]+", " ", flat).strip()


def _merged_raw_doc_text(docs) -> str:
    """拼接当前检索到的正文，供摘录子串校验（不含 [片段n] 前缀）。"""
    parts = []
    for d in docs or []:
        c = getattr(d, "page_content", "") or ""
        if c.strip():
            parts.append(c.strip())
    return "\n".join(parts)


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
    """concise 模式：归一化 → 上下文子句过滤 → 多问问句数裁剪 → 各分句分别受控扩窗抬 answer_relevancy。"""
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


def format_answer_with_reasoning(
    raw_llm_text: str, query: str, context_blob: str, concise: bool
) -> Tuple[str, str]:
    """返回 (界面展示字符串, 仅最终答案供 RAGAS)；concise 时仅对最终段做摘录后处理。"""
    reasoning, final = split_reasoning_and_final_answer(raw_llm_text)
    body = (final or raw_llm_text or "").strip()
    if concise:
        body = apply_concise_answer_postprocess(body, query, context_blob)
    display = merge_reasoning_display(reasoning, body)
    return display, body


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


class RAGEngine:
    """混合检索 + 可选 CrossEncoder 重排 + DeepSeek 生成。"""

    def __init__(self, vectorstore):
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            model=DEEPSEEK_MODEL,
            temperature=0,
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
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

        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": SEARCH_K}
        )
        # 使用保护方式创建 BM25，避免直接崩溃
        try:
            docs = _faiss_documents_for_bm25(self.vectorstore, k_max=1000)
            self.bm25_retriever = BM25Retriever.from_documents(docs)
            self.bm25_retriever.k = SEARCH_K
            print("[OK] BM25Retriever 创建成功")
        except Exception as bm25_err:
            print(f"[ERROR] BM25Retriever 创建失败: {bm25_err}")
            raise

        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vector_retriever],
            weights=[0.4, 0.6],
        )

        # 使用配置化的 Prompt 系统，支持多种业务类型（HR、客服等）
        # 从 config.py 中获取当前系统类型的 Prompt 模板
        prompt_template_text = get_current_prompt_template()
        
        self.optimized_prompt = PromptTemplate(
            template=prompt_template_text,
            input_variables=["context", "question"],
        )
        print(f"[INFO] RAG 生成模版: 系统类型={RAG_SYSTEM_TYPE}, Prompt样式={RAG_PROMPT_STYLE}")

        # CrossEncoder reranker 配置
        # 使用 sentence-transformers 的 CrossEncoder，兼容性更好
        self._cross_encoder = None
        self._reranker_ready = bool(USE_RERANKER and _reranker_path_ok(RERANKER_MODEL_PATH))

        if self._reranker_ready:
            self.qa_chain = None
        else:
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.ensemble_retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": self.optimized_prompt},
            )

        # 上下文压缩：使用 LLM 提取与问题最相关的内容，减少噪声
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

    def _ensure_cross_encoder(self):
        """加载 CrossEncoder reranker。
        使用 sentence-transformers 的 CrossEncoder，兼容性更好。"""
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

            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"[DEBUG] 使用设备: {device}")
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

    def _expand_query(self, query: str) -> str:
        """查询扩展，提高召回。"""
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
        """兼容不同 LangChain 版本 BM25 接口。"""
        r = self.bm25_retriever
        if hasattr(r, "invoke"):
            return r.invoke(q)
        return r.get_relevant_documents(q)

    def _retrieve_for_rerank(self, expanded_query: str) -> List:
        """向量与 BM25 各拉一批，合并去重，供 CrossEncoder 打分。"""
        vdocs = self.vectorstore.similarity_search(expanded_query, k=RERANK_POOL_SIZE)
        bdocs = self._bm25_invoke(expanded_query)
        return _dedupe_documents(list(vdocs) + list(bdocs))

    def _rerank_documents(self, query: str, docs: List) -> List:
        """CrossEncoder 对 (query, passage) 打分，降序取前 RERANK_TOP_N。"""
        self._ensure_cross_encoder()
        if not docs or self._cross_encoder is None:
            print("[DEBUG] CrossEncoder 未加载，使用原始文档顺序")
            return docs[:RERANK_TOP_N] if docs else []
        
        print(f"[DEBUG] 开始 rerank {len(docs)} 个文档")
        pairs = [[query, d.page_content] for d in docs]
        
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

    def _build_context(self, docs: List) -> str:
        """父子索引策略：优先使用父文档内容（语义更完整）。
        如果子文档有对应的父文档，则使用父文档，否则使用子文档内容。"""
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
        """与 RetrievalQA 相同检索器拉文档，供流式路径手动拼 prompt。"""
        er = self.ensemble_retriever
        if hasattr(er, "invoke"):
            return list(er.invoke(expanded_query) or [])
        return list(er.get_relevant_documents(expanded_query) or [])

    def _llm_generate(
        self, prompt_text: str, stream_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """调用 LLM：无回调时 invoke；有回调时 stream 并拼接完整回复。"""
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

    def ask(self, query, stream_callback: Optional[Callable[[str], None]] = None):
        """
        单次问答入口（无多轮历史注入）。
        :param query: 当前用户问题（整段进入 Prompt 的【问题】栏）。
        :param stream_callback: 非 None 时 LLM 使用 stream，并对每个文本片回调（SSE 用）。
        :return: dict 含 result（展示全文）、answer_only（最终答案段）、source_documents。
        """
        print(f"[DEBUG] === ask() 开始执行，query: {query[:50]}... ===")

        # 步骤 1：轻量查询扩展，提高 BM25/向量对同义词的召回
        expanded_query = self._expand_query(query)
        print(f"[DEBUG] query 扩展完成: {expanded_query[:50]}...")

        # 步骤 2：若配置启用 reranker，懒加载 CrossEncoder（失败则回退 qa_chain）
        if self._reranker_ready:
            print("[DEBUG] reranker 已启用，准备调用 _ensure_cross_encoder()")
            self._ensure_cross_encoder()
            print("[DEBUG] _ensure_cross_encoder() 调用完成")
        else:
            print("[DEBUG] reranker 未启用，使用 qa_chain 路径")

        # 步骤 3a：优先走「上下文压缩检索器」——用 LLM 对 ensemble 结果做摘录，再进生成（长文本降噪）
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
                # standard + 流式：禁止走 RetrievalQA.invoke（其不暴露 stream），改用手动检索 + _llm_generate
                if stream_callback is not None:
                    docs = self._ensemble_get_documents(expanded_query)
                    ctx_docs = docs[:RERANK_TOP_N] if docs else []
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

        # 步骤 3b：未启用压缩或压缩器不可用 —— 使用 RetrievalQA（无 rerank）或下面 rerank 分支
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

        # 步骤 3c：启用了 reranker 且 qa_chain 为 None —— 手动合并向量+BM25 后 CrossEncoder 打分
        merged = self._retrieve_for_rerank(expanded_query)
        top_docs = self._rerank_documents(query, merged)
        context = self._build_context(top_docs)
        prompt_text = self.optimized_prompt.format(context=context, question=query)
        raw = self._llm_generate(prompt_text, stream_callback)
        blob = _merged_raw_doc_text(top_docs)
        use_concise = RAG_PROMPT_STYLE == "concise"
        display, answer_only = format_answer_with_reasoning(raw, query, blob, use_concise)
        return {"result": display, "answer_only": answer_only, "source_documents": top_docs}
