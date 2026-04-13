# -*- coding: utf-8 -*-
"""
HR RAG HTTP 服务（FastAPI）
==========================
【职责】
  - 生命周期内初始化 MySQL（可选）、KnowledgeBase、RAGEngine（见 _bootstrap_rag_core）
  - REST：健康检查、会话列表/历史、非流式与 SSE 流式问答
  - 启用 DATABASE_URL 时：每条问答写入 rag_messages（见 db/chat_store）

【记忆】落库仅用于前端展示与审计；RAGEngine.ask 仍只接收当前 question，不把历史拼进 Prompt。

启动：uvicorn api_server:app --host 0.0.0.0 --port 8000  或  python api_server.py

前端：POST /api/chat 或 /api/chat/stream；SSE 帧为 data: {json}\\n\\n；跨域配 API_CORS_ORIGINS。

【阅读顺序】
  1) `_lifespan`：启动时 init_database → KnowledgeBase → `_bootstrap_rag_core` → 可选 `_ensure_cross_encoder`
  2) `chat`：落库用户句 → `to_thread(ask)` → 序列化 sources → 落库助手句
  3) `_chat_stream_events`：同上但 `ask(..., stream_callback)` + 队列桥接线程与协程

【为何大量 to_thread】RAGEngine.ask 内部为同步阻塞（向量检索、CrossEncoder、HTTP LLM），在 async 路由里必须放到线程池，避免阻塞整个事件循环。
"""
import asyncio  # 异步队列与线程配合做 SSE
import json  # SSE 与 JSON 响应序列化
import os  # 须最先设置线程相关环境变量（与 main 一致）
from urllib.parse import quote  # 制度文档外链路径编码
import sys  # 可选：保证 UTF-8 输出
import threading  # 在后台线程跑同步的 engine.ask，避免阻塞事件循环
from contextlib import asynccontextmanager  # FastAPI 生命周期里加载向量库
from typing import Any, AsyncGenerator, Dict, List, Optional  # 类型标注

# 与 main.py 相同：在 import torch/langchain 前固定单线程，降低 Windows 下 native 冲突
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_MAX_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import io

# Windows 控制台尽量 UTF-8（日志）
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    except Exception:
        pass

from fastapi import Depends, FastAPI, Header, HTTPException  # Web 与依赖注入
from fastapi.middleware.cors import CORSMiddleware  # 浏览器跨域
from fastapi.responses import StreamingResponse  # SSE 流式响应
from pydantic import BaseModel, Field  # 请求体验证

# ========== 配置导入（热路径仅 FAISS 路径、CORS、DB 开关等；RAG 细节在 rag_engine 读 config）==========
from config import (  # 统一配置
    API_CORS_ORIGINS,
    API_HOST,
    API_PORT,
    DATABASE_ENABLED,
    DOCS_PUBLIC_BASE_URL,
    FAISS_INDEX_PATH,
    RAG_USE_LANGGRAPH,
    USE_RERANKER,
    get_seed_document_path,
)

# ========== 进程级单例：由 lifespan 赋值，请求处理阶段只读（无锁；uvicorn 单 worker 假设）==========
_kb_shared: Any = None  # 嵌入模型与 FAISS 加载入口
_rag_engine: Any = None  # 每个 HTTP 问答最终调用其 ask()


def _get_user_id_dep(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> int:
    """解析调用方用户；未配置数据库时返回 0。"""
    from db.auth import resolve_user_id

    return resolve_user_id(x_api_key)


def _serialize_sources(docs: Optional[List], max_items: int = 5, preview_len: int = 240) -> List[Dict[str, Any]]:
    """
    把 `source_documents` 转成 JSON 可序列化列表：preview、label（文件名）、href（配了 DOCS_PUBLIC_BASE_URL 时）。
    与 `chat_store.persist_assistant_answer` 写入的 extra.sources 结构一致，供前端「参考来源」展示。
    """
    out: List[Dict[str, Any]] = []
    if not docs:
        return out
    for d in docs[:max_items]:
        raw = getattr(d, "page_content", "") or ""
        one_line = raw.replace("\n", " ").strip()
        preview = one_line[:preview_len] + ("..." if len(one_line) > preview_len else "")
        meta = getattr(d, "metadata", None) or {}
        meta_dict = dict(meta) if isinstance(meta, dict) else {}
        src_path = meta_dict.get("source") or ""
        label = os.path.basename(str(src_path)) if src_path else ""
        if not label:
            label = "制度摘录"
        href: Optional[str] = None
        if DOCS_PUBLIC_BASE_URL and src_path:
            bn = os.path.basename(str(src_path))
            if bn:
                href = f"{DOCS_PUBLIC_BASE_URL}/{quote(bn)}"
        out.append(
            {
                "preview": preview,
                "label": label,
                "href": href,
                "metadata": meta_dict,
            }
        )
    return out


def _fallback_demo_vectorstore(kb) -> Any:
    """无可用索引时按配置解析种子文档并建库（与 main.py 一致）。"""
    file_path = get_seed_document_path()
    print(f"[API] 使用种子文档: {file_path}")
    texts = kb.process_document(file_path)
    return kb.create_vector_store(texts)


def _bootstrap_rag_core(kb) -> Any:
    """
    启动时：若 FAISS_INDEX_PATH 已有索引则加载；否则用种子文档（与 main.py 一致）建库。
    """
    from rag_engine import RAGEngine

    if os.path.exists(FAISS_INDEX_PATH):
        vs = kb.load_vector_store()
        if vs is not None:
            print("[API] 已加载本地 FAISS 索引")
            return RAGEngine(vs)
    vs = _fallback_demo_vectorstore(kb)
    print("[API] 已用种子文档初始化索引")
    return RAGEngine(vs)


# ========== FastAPI 生命周期：唯一构造 RAGEngine 的位置（与 main.py CLI 独立）==========


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """应用启动：MySQL 建表 + 单例 KnowledgeBase + RAGEngine。"""
    global _rag_engine, _kb_shared
    from db.database import init_database
    from knowledge_base import KnowledgeBase

    init_database()
    _kb_shared = KnowledgeBase()
    _rag_engine = await asyncio.to_thread(_bootstrap_rag_core, _kb_shared)
    # 启动时预加载 CrossEncoder，避免首个请求额外多秒冷启动（仅 USE_RERANKER=true 时）
    if USE_RERANKER and _rag_engine is not None:
        await asyncio.to_thread(_rag_engine._ensure_cross_encoder)
    if RAG_USE_LANGGRAPH and _rag_engine is not None:
        await asyncio.to_thread(_rag_engine._get_agentic_graph)
        print("[API] RAG_USE_LANGGRAPH=true，问答走 LangGraph Agentic 路径")
    print("[API] RAG 引擎已就绪")
    yield  # 应用运行中；关闭时清理引用，便于多进程 reload 场景释放句柄
    _rag_engine = None
    _kb_shared = None


app = FastAPI(title="HR RAG API", version="1.0.0", lifespan=_lifespan)

# CORS：前端独立域名/端口时必须开启
_cors = API_CORS_ORIGINS if API_CORS_ORIGINS else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """聊天请求：问题正文 + 可选会话 id（多轮时由前端回传）。"""

    question: str = Field(..., min_length=1, max_length=16000)
    conversation_id: Optional[str] = Field(None, max_length=36)


class ChatResponse(BaseModel):
    """非流式响应：结果 + 来源 + 当前会话 id（启用 MySQL 时）。"""

    result: str
    answer_only: str
    sources: List[Dict[str, Any]]
    conversation_id: Optional[str] = None


def _sse_data(obj: Dict[str, Any]) -> str:
    """SSE 文本帧：前缀 data: + JSON + 双换行（EventSource 约定）。"""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# ---------- 探活 ----------
@app.get("/health")
async def health():
    """负载均衡/容器探活：不触发推理；含数据库连通性。"""
    from db.database import ping_database

    rag_ok = _rag_engine is not None
    db_ok: Optional[bool] = None
    if DATABASE_ENABLED:
        db_ok = await asyncio.to_thread(ping_database)
    return {"ok": rag_ok, "database": db_ok}


# ---------- 会话持久化（MySQL）----------
@app.get("/api/conversations")
async def api_list_conversations(
    user_id: int = Depends(_get_user_id_dep),
    limit: int = 50,
):
    """当前用户的会话列表（需配置 DATABASE_URL）。"""
    if not DATABASE_ENABLED or user_id == 0:
        return []
    from db.chat_store import list_conversations
    from db.database import SessionLocal

    def _run():
        with SessionLocal() as session:
            return list_conversations(session, user_id, min(limit, 200))

    return await asyncio.to_thread(_run)


@app.get("/api/conversations/{conversation_id}/messages")
async def api_get_messages(conversation_id: str, user_id: int = Depends(_get_user_id_dep)):
    """拉取会话历史消息，供前端恢复多轮上下文展示。"""
    if not DATABASE_ENABLED or user_id == 0:
        raise HTTPException(status_code=404, detail="未启用数据库持久化")
    from db.chat_store import list_messages
    from db.database import SessionLocal

    def _run():
        with SessionLocal() as session:
            return list_messages(session, user_id, conversation_id)

    try:
        return await asyncio.to_thread(_run)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


# ---------- 问答：核心为 _rag_engine.ask；DB 仅审计/前端历史，不参与 Prompt ----------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_id: int = Depends(_get_user_id_dep)):
    """
    非流式问答。
    顺序：鉴权 user_id →（可选）落库用户句 → **线程池** `ask`（CPU/GPU/HTTP 阻塞在此线程）→ 序列化 sources →（可选）落库助手句。
    """
    if _rag_engine is None:
        raise HTTPException(status_code=503, detail="RAG engine not ready")
    q = req.question.strip()
    conv_id: Optional[str] = req.conversation_id

    # 先写用户消息并确定 conversation_id（新会话则 UUID）
    if DATABASE_ENABLED and user_id:
        from db.chat_store import persist_user_question
        from db.database import SessionLocal

        def _user_turn():
            with SessionLocal() as session:
                return persist_user_question(session, user_id, req.conversation_id, q)

        try:
            conv_id = await asyncio.to_thread(_user_turn)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    # 同步 RAG（在线程池执行，避免阻塞事件循环）；可选 LangGraph Agentic 路径
    try:
        if RAG_USE_LANGGRAPH:
            raw = await asyncio.to_thread(_rag_engine.ask_agentic, q)
        else:
            raw = await asyncio.to_thread(_rag_engine.ask, q)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    result = str(raw.get("result", "") or "")
    ans_only = str(raw.get("answer_only", "") or raw.get("result", "") or "")
    src = _serialize_sources(raw.get("source_documents"))

    # 再写助手消息（完整 result + 摘要字段）
    if DATABASE_ENABLED and user_id and conv_id:
        from db.chat_store import persist_assistant_answer
        from db.database import SessionLocal

        def _asst():
            with SessionLocal() as session:
                persist_assistant_answer(session, conv_id, result, ans_only, src)

        try:
            await asyncio.to_thread(_asst)
        except Exception as ex:
            print(f"[WARN] 助手回复落库失败: {ex}")

    return ChatResponse(
        result=result,
        answer_only=ans_only,
        sources=src,
        conversation_id=conv_id if DATABASE_ENABLED and user_id else None,
    )


# ---------- SSE：工作线程跑同步 ask + stream_callback；主协程 await Queue 写 event-stream ----------
async def _chat_stream_events(
    question: str,
    user_id: int,
    conversation_id_in: Optional[str],
) -> AsyncGenerator[str, None]:
    """
    流式问答生成器：产出 SSE 字符串帧。
    线程模型：`worker` 线程跑同步 `ask(stream_callback=on_token)`；`on_token` 用 call_soon_threadsafe 把片段塞进 asyncio.Queue；
    主协程 `await aq.get()` 再 `yield`，避免在 worker 里直接操作异步上下文。
    帧类型：meta（conversation_id）→ 可选多条 status（阶段提示）→ 多条 delta → 一条 final（完整 result + sources）。
    """
    if _rag_engine is None:
        yield _sse_data({"t": "error", "d": "RAG engine not ready"})
        return

    conv_id: Optional[str] = conversation_id_in
    # 流式同样先落库用户轮次，便于首帧带回 conversation_id
    if DATABASE_ENABLED and user_id:
        from db.chat_store import persist_user_question
        from db.database import SessionLocal

        def _user_turn():
            with SessionLocal() as session:
                return persist_user_question(session, user_id, conversation_id_in, question)

        try:
            conv_id = await asyncio.to_thread(_user_turn)
        except ValueError as e:
            yield _sse_data({"t": "error", "d": str(e)})
            return
        yield _sse_data({"t": "meta", "d": {"conversation_id": conv_id}})

    loop = asyncio.get_running_loop()
    aq: asyncio.Queue = asyncio.Queue()

    def on_token(t: str) -> None:
        # 从工作线程安全投递到 asyncio 队列
        loop.call_soon_threadsafe(aq.put_nowait, ("delta", t))

    def on_status(s: str) -> None:
        # 检索/质检/生成前等阶段文案，先于首条 delta 到达，缓解「长时间无输出」观感
        loop.call_soon_threadsafe(aq.put_nowait, ("status", s))

    def worker() -> None:
        try:
            if RAG_USE_LANGGRAPH:
                r = _rag_engine.ask_agentic(
                    question, stream_callback=on_token, status_callback=on_status
                )
            else:
                r = _rag_engine.ask(
                    question, stream_callback=on_token, status_callback=on_status
                )
            if DATABASE_ENABLED and user_id and conv_id:
                from db.chat_store import persist_assistant_answer
                from db.database import SessionLocal

                try:
                    with SessionLocal() as session:
                        persist_assistant_answer(
                            session,
                            conv_id,
                            str(r.get("result", "") or ""),
                            str(r.get("answer_only", "") or r.get("result", "") or ""),
                            _serialize_sources(r.get("source_documents")),
                        )
                except Exception as ex:
                    print(f"[WARN] 流式助手回复落库失败: {ex}")
            loop.call_soon_threadsafe(aq.put_nowait, ("final", r))
        except Exception as e:
            loop.call_soon_threadsafe(aq.put_nowait, ("err", str(e)))

    threading.Thread(target=worker, daemon=True).start()

    while True:
        kind, payload = await aq.get()  # 阻塞直到 delta / final / err
        if kind == "delta":
            yield _sse_data({"t": "delta", "d": payload})
        elif kind == "status":
            yield _sse_data({"t": "status", "d": payload})
        elif kind == "final":
            r = payload or {}
            yield _sse_data(
                {
                    "t": "final",
                    "d": {
                        "result": str(r.get("result", "") or ""),
                        "answer_only": str(r.get("answer_only", "") or r.get("result", "") or ""),
                        "sources": _serialize_sources(r.get("source_documents")),
                        "conversation_id": conv_id if DATABASE_ENABLED and user_id else None,
                    },
                }
            )
            break
        elif kind == "err":
            yield _sse_data({"t": "error", "d": payload})
            break


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, user_id: int = Depends(_get_user_id_dep)):
    """
    流式问答：text/event-stream。
    t=meta：{conversation_id}；t=status 阶段文案；t=delta；t=final（含 conversation_id）；t=error。
    """
    q = req.question.strip()
    return StreamingResponse(
        _chat_stream_events(q, user_id, req.conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn  # 仅直接 python api_server.py 时使用

    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=False)
