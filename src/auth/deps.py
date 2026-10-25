"""FastAPI 鉴权依赖：从 Authorization header 取 JWT，验证并返回当前用户"""
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from src.auth.jwt import decode_token
from src.db.repository import UserRepository

# 从 Authorization: Bearer <token> 提取 token
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict:
    """验证 JWT，返回当前用户信息 {id, username, role}

    无效或过期抛出 401。
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="无效或过期的凭证",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    user_id: Optional[str] = payload.get("sub")
    role: str = payload.get("role", "user")
    if user_id is None:
        raise credentials_exception

    # 从数据库确认用户存在（防止 token 有效但用户已被删除）
    user = UserRepository.get_by_id(user_id)
    if user is None:
        raise credentials_exception

    return {
        "id": user["id"],
        "username": user["username"],
        "role": user["role"],
    }
