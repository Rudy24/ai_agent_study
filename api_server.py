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
"""
import asyncio  # 异步队列与线程配合做 SSE
import json  # SSE 与 JSON 响应序列化
import os  # 须最先设置线程相关环境变量（与 main 一致）
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

from config import (  # 统一配置
    API_CORS_ORIGINS,
    API_HOST,
    API_PORT,
    DATABASE_ENABLED,
    FAISS_INDEX_PATH,
    get_seed_document_path,
)

# 全局：共享 KnowledgeBase（复用 Embedding）、RAG 引擎
_kb_shared: Any = None
_rag_engine: Any = None


def _get_user_id_dep(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> int:
    """解析调用方用户；未配置数据库时返回 0。"""
    from db.auth import resolve_user_id

    return resolve_user_id(x_api_key)


def _serialize_sources(docs: Optional[List], max_items: int = 5, preview_len: int = 240) -> List[Dict[str, Any]]:
    """把 LangChain Document 列表转成可 JSON 序列化的来源列表（供前端展示摘要）。"""
    out: List[Dict[str, Any]] = []
    if not docs:
        return out
    for d in docs[:max_items]:
        raw = getattr(d, "page_content", "") or ""
        one_line = raw.replace("\n", " ").strip()
        preview = one_line[:preview_len] + ("..." if len(one_line) > preview_len else "")
        meta = getattr(d, "metadata", None) or {}
        out.append({"preview": preview, "metadata": dict(meta) if isinstance(meta, dict) else {}})
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


@asynccontextmanager
async def _lifespan(app: FastAPI):
    """应用启动：MySQL 建表 + 单例 KnowledgeBase + RAGEngine。"""
    global _rag_engine, _kb_shared
    from db.database import init_database
    from knowledge_base import KnowledgeBase

    init_database()
    _kb_shared = KnowledgeBase()
    _rag_engine = await asyncio.to_thread(_bootstrap_rag_core, _kb_shared)
    print("[API] RAG 引擎已就绪")
    yield
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


# ---------- 问答（核心：调用 RAGEngine，与 DB 解耦）----------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, user_id: int = Depends(_get_user_id_dep)):
    """非流式问答；启用 MySQL 时写入用户问题与助手回答。"""
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

    # 同步 RAG（在线程池执行，避免阻塞事件循环）
    try:
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


# ---------- SSE：线程内 ask(stream_callback)，主协程从队列吐帧 ----------
async def _chat_stream_events(
    question: str,
    user_id: int,
    conversation_id_in: Optional[str],
) -> AsyncGenerator[str, None]:
    """流式问答 + 可选落库：首包 meta 携带 conversation_id。"""
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

    def worker() -> None:
        try:
            r = _rag_engine.ask(question, stream_callback=on_token)
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
    t=meta：{conversation_id}；t=delta；t=final（含 conversation_id）；t=error。
    """
    q = req.question.strip()
    return StreamingResponse(
        _chat_stream_events(q, user_id, req.conversation_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn  # 仅直接 python api_server.py 时使用

    uvicorn.run("api_server:app", host=API_HOST, port=API_PORT, reload=False)
