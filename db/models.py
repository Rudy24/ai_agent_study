# -*- coding: utf-8 -*-
"""
SQLAlchemy ORM 模型
==================
【表】rag_users → rag_conversations → rag_messages（一对多级联删除）
【用途】聊天 UI 持久化与审计；不参与 RAG 检索，也不向 LLM 注入历史（见 ARCHITECTURE 4.5）。
"""
from __future__ import annotations  # 延迟注解，兼容 3.9

from datetime import datetime, timezone  # UTC 时间
from typing import Optional  # 可选类型

from sqlalchemy import DateTime, ForeignKey, Integer, JSON, String, Text  # 列类型
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship  # 2.0 风格 ORM


def utc_now():
    """数据库默认值用的当前 UTC 时间。"""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """声明式基类。"""


class User(Base):
    """调用方身份：可与 X-API-Key 的 SHA256 绑定；无 Key 模式使用 DEFAULT_APP_USER 行。"""

    __tablename__ = "rag_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    api_key_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)  # 明文 Key 不落库
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    conversations: Mapped[list["Conversation"]] = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    """会话容器：id 为 UUID 字符串；title 通常取首问截断。"""

    __tablename__ = "rag_conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    user_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("rag_users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)  # 列表展示用
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    user: Mapped["User"] = relationship("User", back_populates="conversations")
    messages: Mapped[list["Message"]] = relationship(
        "Message",
        back_populates="conversation",
        order_by="Message.created_at",
        cascade="all, delete-orphan",
    )


class Message(Base):
    """一条消息：role 为 user / assistant；assistant 的 extra 可存 source_count 等 JSON。"""

    __tablename__ = "rag_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("rag_conversations.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)  # 助手存完整 result 展示文本
    extra: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    conversation: Mapped["Conversation"] = relationship("Conversation", back_populates="messages")
