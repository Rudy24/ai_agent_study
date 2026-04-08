# -*- coding: utf-8 -*-
"""
会话与消息持久化 API
==================
供 api_server 在线程中调用：写入用户/助手轮次；列表接口供前端恢复气泡。
【注意】此处不修改 RAG 行为；历史 content 不会自动拼入 RAGEngine.ask。

【调用关系】
  - 写：`persist_user_question` →（RAG）→ `persist_assistant_answer`，由 `api_server` 的 `to_thread` 包一层 DB 会话。
  - 读：`list_conversations` / `list_messages` 供 REST 列表与拉历史。
"""
import uuid  # 新会话 id
from datetime import datetime, timezone  # 更新时间戳
from typing import Any, Dict, List, Optional  # 类型

from sqlalchemy.orm import Session  # ORM 会话

from db.models import Conversation, Message  # 模型


# ---------- 写路径：一轮问答对应「用户 Message」+ 可选新建 Conversation ----------


def persist_user_question(
    session: Session,
    user_id: int,
    conversation_id: Optional[str],
    question: str,
) -> str:
    """
    写入用户问题；无 conversation_id 时新建会话（UUID），title 取首问前 200 字。
    校验：已有 id 必须属于该 user_id，否则 ValueError（防越权）。
    返回当前会话 id，供后续 `persist_assistant_answer` 与 API 响应回传前端。
    """
    now = datetime.now(timezone.utc)
    if conversation_id:
        conv = (
            session.query(Conversation)
            .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
            .first()
        )
        if not conv:
            raise ValueError("会话不存在或无权访问")
        conv.updated_at = now
    else:
        conversation_id = str(uuid.uuid4())
        title = question[:200] + ("…" if len(question) > 200 else "")
        conv = Conversation(id=conversation_id, user_id=user_id, title=title)
        session.add(conv)
    session.add(
        Message(
            conversation_id=conversation_id,
            role="user",
            content=question,
            extra=None,
        )
    )
    session.commit()
    return conversation_id


def persist_assistant_answer(
    session: Session,
    conversation_id: str,
    result: str,
    answer_only: str,
    sources: Optional[List[Dict[str, Any]]],
) -> None:
    """
    写入助手消息：`content` 存完整展示文本（含推理段）；`extra` 存 answer 摘要、来源条数与最多 10 条 sources。
    用于前端「参考来源」回放与审计，不参与下次 RAG 检索。
    """
    now = datetime.now(timezone.utc)
    conv = session.query(Conversation).filter(Conversation.id == conversation_id).first()
    if conv:
        conv.updated_at = now
    extra: Dict[str, Any] = {
        "answer_only_preview": (answer_only or "")[:2000],
        "source_count": len(sources or []),
        "sources": (sources or [])[:10],
    }
    session.add(
        Message(
            conversation_id=conversation_id,
            role="assistant",
            content=result or "",
            extra=extra,
        )
    )
    session.commit()


# ---------- 读路径：列表与单会话时间序 ----------


def list_conversations(session: Session, user_id: int, limit: int = 50) -> List[Dict[str, Any]]:
    """最近会话列表（摘要）。"""
    rows = (
        session.query(Conversation)
        .filter(Conversation.user_id == user_id)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "title": r.title,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "updated_at": r.updated_at.isoformat() if r.updated_at else None,
        }
        for r in rows
    ]


def list_messages(session: Session, user_id: int, conversation_id: str) -> List[Dict[str, Any]]:
    """单会话消息时间序；校验归属。"""
    conv = (
        session.query(Conversation)
        .filter(Conversation.id == conversation_id, Conversation.user_id == user_id)
        .first()
    )
    if not conv:
        raise ValueError("会话不存在或无权访问")
    rows = (
        session.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc())
        .all()
    )
    return [
        {
            "id": r.id,
            "role": r.role,
            "content": r.content,
            "extra": r.extra,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]
