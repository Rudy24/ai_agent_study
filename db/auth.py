# -*- coding: utf-8 -*-
"""根据 X-API-Key 解析用户 id；未启用数据库时返回 0（不落库）。"""
import hashlib  # 校验 API Key
from typing import Optional  # 类型

from fastapi import HTTPException  # 401

from config import (  # 开关与默认用户
    API_REQUIRE_API_KEY,
    API_SERVICE_API_KEY,
    DATABASE_ENABLED,
    DEFAULT_APP_USER,
)
from db.database import SessionLocal  # 会话
from db.models import User  # 用户表


def resolve_user_id(x_api_key: Optional[str]) -> int:
    """
    返回 rag_users.id。
    - 未启用数据库：0（调用方应跳过持久化）
    - API_REQUIRE_API_KEY：必须带 Key 且匹配库中用户
    - 否则：有 Key 则匹配；无 Key 则用 DEFAULT_APP_USER
    """
    if not DATABASE_ENABLED or SessionLocal is None:
        return 0
    with SessionLocal() as session:
        # 生产强制鉴权：必须带 X-API-Key 且与某用户的 api_key_hash 匹配
        if API_REQUIRE_API_KEY:
            if not x_api_key:
                raise HTTPException(status_code=401, detail="缺少请求头 X-API-Key")
            digest = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
            u = session.query(User).filter(User.api_key_hash == digest).first()
            if not u:
                raise HTTPException(status_code=401, detail="无效的 X-API-Key")
            return u.id
        # 开发友好：带了 Key 则尝试匹配；否则回落到默认业务用户（须已在 init_database 创建）
        if x_api_key:
            digest = hashlib.sha256(x_api_key.encode("utf-8")).hexdigest()
            u = session.query(User).filter(User.api_key_hash == digest).first()
            if u:
                return u.id
        u = session.query(User).filter(User.username == DEFAULT_APP_USER).first()
        if not u:
            raise HTTPException(status_code=503, detail="数据库未初始化默认用户，请检查启动日志")
        return u.id
