"""JWT 签发与验证"""
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt

from src.config import settings


def create_access_token(
    subject: str,
    role: str = "user",
    expires_minutes: Optional[int] = None,
) -> str:
    """签发 JWT

    Args:
        subject: 用户 id（sub 字段）
        role: 用户角色
        expires_minutes: 过期分钟数，默认用配置
    """
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.jwt_expire_minutes
    )
    payload = {
        "sub": subject,
        "role": role,
        "exp": expire,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[dict]:
    """验证 JWT，返回 payload（sub, role）或 None（无效/过期）"""
    try:
        payload = jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
        return payload
    except JWTError:
        return None
