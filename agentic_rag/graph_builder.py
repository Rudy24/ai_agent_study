"""
LangGraph Agentic RAG 状态机
----------------------------
流程：prepare → retrieve →（可选）grade → rewrite → retrieve 循环（有上限）→ generate。
与经典 `RAGEngine.ask` 共用同一套向量/BM25/CrossEncoder 与生成后处理。
"""

from __future__ import annotations

import json  # 解析质检 JSON
from typing import Any, Callable, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage  # 质检与改写用的单条用户消息
from langchain_core.runnables import RunnableConfig  # 节点第二参，承载 configurable
from langgraph.graph import END, START, StateGraph  # 有向图与起止

from config import RAG_AGENT_GRADE_ENABLED, RAG_AGENT_MAX_RETRIEVE_ROUNDS


def _status_from_config(config: RunnableConfig) -> Optional[Callable[[str], None]]:
    """从 invoke 传入的 configurable 取出 status_callback。"""
    return (config.get("configurable") or {}).get("status_callback")


class AgenticRAGState(TypedDict, total=False):
    """跨节点传递的字段；total=False 允许逐步合并。"""

    question: str
    retrieval_query: str
    documents: List[Any]
    retrieval_round: int
    grade_ok: bool
    grade_reason: str
    final_payload: Dict[str, Any]


def build_agentic_rag_graph(engine: Any):
    """
    将已有 RAGEngine 实例绑定进各节点闭包，编译为可 `invoke` 的图。
    :param engine: `rag_engine.RAGEngine` 实例（需含 retrieve_top_documents、_expand_query、llm、_answer_from_context_docs）。
    """

    def node_prepare(state: AgenticRAGState, config: RunnableConfig) -> Dict[str, Any]:
        """用引擎既有同义词扩展初始化检索串；检索轮次归零。"""
        sc = _status_from_config(config)
        if sc:
            sc("正在解析与扩展检索词…")
        q = (state.get("question") or "").strip()
        rq = engine._expand_query(q)
        return {"retrieval_query": rq, "retrieval_round": 0}

    def node_retrieve(state: AgenticRAGState, config: RunnableConfig) -> Dict[str, Any]:
        """按当前 retrieval_query 召回，精排仍用用户原 question。"""
        sc = _status_from_config(config)
        if sc:
            sc("正在检索知识库并重排…")
        rq = (state.get("retrieval_query") or "").strip() or engine._expand_query(
            state["question"]
        )
        q = state["question"]
        docs = engine.retrieve_top_documents(rq, q)
        prev = int(state.get("retrieval_round") or 0)
        return {"documents": docs, "retrieval_round": prev + 1}

    def node_grade(state: AgenticRAGState, config: RunnableConfig) -> Dict[str, Any]:
        """LLM 判断当前片段是否足以支撑制度问答；解析失败时默认通过，避免卡死。"""
        sc = _status_from_config(config)
        if sc:
            sc("正在评估检索结果是否充分…")
        q = state["question"]
        docs = state.get("documents") or []
        snippets: List[str] = []
        for d in docs[:5]:
            pc = getattr(d, "page_content", "") or ""
            snippets.append(pc[:280].replace("\n", " "))
        blob = "\n---\n".join(snippets) if snippets else "（无检索结果）"
        prompt = (
            "判断下列检索片段整体是否足以回答用户关于公司人事制度的问题"
            "（能否从片段中找到直接或间接依据）。\n"
            "只输出一个 JSON 对象，键 sufficient (true/false) 与 reason (简短字符串)。\n"
            f"用户问题：{q}\n检索片段摘要：\n{blob}"
        )
        resp = engine.llm.invoke([HumanMessage(content=prompt)])
        raw = resp.content if hasattr(resp, "content") else str(resp)
        ok = True
        reason = ""
        try:
            s, e = raw.find("{"), raw.rfind("}")
            if s >= 0 and e > s:
                j = json.loads(raw[s : e + 1])
                ok = bool(j.get("sufficient"))
                reason = str(j.get("reason", ""))[:400]
        except Exception:
            ok = True
            reason = "grade_parse_fallback"
        return {"grade_ok": ok, "grade_reason": reason}

    def node_rewrite(state: AgenticRAGState, config: RunnableConfig) -> Dict[str, Any]:
        """根据质检原因生成更适合制度库检索的查询句。"""
        sc = _status_from_config(config)
        if sc:
            sc("正在改写检索查询…")
        q = state["question"]
        reason = (state.get("grade_reason") or "片段与问题匹配不足").strip()
        prompt = (
            "你是企业制度检索助手。根据用户问题和检索不足的原因，"
            "改写出一条更利于在制度知识库中召回的检索查询（一句话，不要解释）。\n"
            f"用户问题：{q}\n原因：{reason}\n仅输出改写后的检索句："
        )
        resp = engine.llm.invoke([HumanMessage(content=prompt)])
        text = (resp.content if hasattr(resp, "content") else str(resp)).strip()
        if len(text) > 800:
            text = text[:800]
        return {"retrieval_query": text}

    def node_generate(state: AgenticRAGState, config: RunnableConfig) -> Dict[str, Any]:
        """拼 Prompt 调用 LLM；stream/status 回调经 configurable 注入。"""
        conf = config.get("configurable") or {}
        cb = conf.get("stream_callback")
        st = conf.get("status_callback")
        docs = state.get("documents") or []
        payload = engine._answer_from_context_docs(state["question"], docs, cb, st)
        return {"final_payload": payload}

    def route_after_retrieve(state: AgenticRAGState) -> Literal["grade", "generate"]:
        """关质检时直接进入生成，省一次 LLM。"""
        if not RAG_AGENT_GRADE_ENABLED:
            return "generate"
        return "grade"

    def route_after_grade(state: AgenticRAGState) -> Literal["rewrite", "generate"]:
        """通过质检或已达最大检索轮次则生成；否则改写后再检索。"""
        if state.get("grade_ok"):
            return "generate"
        rnd = int(state.get("retrieval_round") or 0)
        if rnd >= RAG_AGENT_MAX_RETRIEVE_ROUNDS:
            return "generate"
        return "rewrite"

    g = StateGraph(AgenticRAGState)
    g.add_node("prepare", node_prepare)
    g.add_node("retrieve", node_retrieve)
    g.add_node("grade", node_grade)
    g.add_node("rewrite", node_rewrite)
    g.add_node("generate", node_generate)

    g.add_edge(START, "prepare")
    g.add_edge("prepare", "retrieve")
    g.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"grade": "grade", "generate": "generate"},
    )
    g.add_conditional_edges(
        "grade",
        route_after_grade,
        {"rewrite": "rewrite", "generate": "generate"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_edge("generate", END)

    return g.compile()
