# -*- coding: utf-8 -*-
"""
数据库连接与会话工厂
====================
【职责】按 DATABASE_URL 创建 SQLAlchemy engine；模块导入时 _setup_engine()；
       init_database() 在 API  lifespan 中调用：建表 + 插入 DEFAULT_APP_USER（及可选 API Key 哈希）。
【注意】未配置 DATABASE_URL 时 engine 为 None，SessionLocal 为 None，全站按「不落库」运行。
"""
import hashlib  # API Key 摘要

from sqlalchemy import create_engine, text  # 引擎与原始 SQL
from sqlalchemy.orm import sessionmaker  # 会话工厂

from config import (  # 配置
    API_SERVICE_API_KEY,
    DATABASE_ENABLED,
    DATABASE_URL,
    DEFAULT_APP_USER,
)
from db.models import Base, User  # 元数据与模型

engine = None  # 未启用数据库时为 None
SessionLocal = None  # 同上


def _setup_engine() -> None:
    """根据 DATABASE_URL 创建连接池（仅启用时）。"""
    global engine, SessionLocal
    if not DATABASE_ENABLED:
        return
    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=3600,
        echo=False,
    )
    SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, expire_on_commit=False)


_setup_engine()


# ---------- 生命周期：API lifespan 首调；与业务请求中的 SessionLocal 分离 ----------


def init_database() -> None:
    """创建表并写入默认用户（幂等）；仅在 DATABASE_ENABLED 时执行。"""
    if not DATABASE_ENABLED or engine is None:
        print("[DB] 未配置 DATABASE_URL，跳过 MySQL 初始化")
        return
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        existing = session.query(User).filter(User.username == DEFAULT_APP_USER).first()
        if existing:
            print(f"[DB] 已存在默认用户: {DEFAULT_APP_USER}")
            return
        h = None
        if API_SERVICE_API_KEY:
            h = hashlib.sha256(API_SERVICE_API_KEY.encode("utf-8")).hexdigest()
        u = User(username=DEFAULT_APP_USER, api_key_hash=h)
        session.add(u)
        session.commit()
        print(f"[DB] 已创建默认用户: {DEFAULT_APP_USER}")


def ping_database() -> bool:
    """健康检查用 SELECT 1。"""
    if not DATABASE_ENABLED or engine is None:
        return False
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
