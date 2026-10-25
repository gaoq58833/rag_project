"""密码哈希：bcrypt（直接使用 bcrypt 库，避免 passlib 兼容问题）"""
import bcrypt


def hash_password(password: str) -> str:
    """明文密码 → bcrypt 哈希"""
    # bcrypt 只处理前 72 字节，先编码再截断
    password_bytes = password.encode("utf-8")[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """校验明文密码与哈希是否匹配"""
    password_bytes = plain_password.encode("utf-8")[:72]
    hashed_bytes = hashed_password.encode("utf-8")
    return bcrypt.checkpw(password_bytes, hashed_bytes)
