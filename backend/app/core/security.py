"""09-用户系统与数据隔离：密码哈希 + JWT（对齐 full-stack-fastapi-template）。

- 哈希：pwdlib（Argon2 优先，Bcrypt 兜底），算法与模板一致。
- 令牌：PyJWT HS256，密钥来自 ``config.security.secret_key``（环境变量注入）。
- ``get_secret_key()`` 在密钥缺失时启动失败（需求 S3），由 ``main.lifespan`` 调用 eager 校验。
"""
from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from pwdlib import PasswordHash
from pwdlib.hashers.argon2 import Argon2Hasher
from pwdlib.hashers.bcrypt import BcryptHasher

from app.config import config

password_hash = PasswordHash((Argon2Hasher(), BcryptHasher()))
ALGORITHM = "HS256"


def get_secret_key() -> str:
    """返回 JWT 密钥；缺失即抛 RuntimeError（由 lifespan eager 捕获 → 启动失败）。"""
    key = (config.get("security", {}) or {}).get("secret_key") or ""
    if not key:
        raise RuntimeError(
            "security.secret_key 未配置：请在 config.yaml 写 ${AUTH_SECRET_KEY} 并在 .env 注入 AUTH_SECRET_KEY"
        )
    return key


def create_access_token(subject: int | str, expires_delta: timedelta) -> str:
    expire = datetime.now(UTC) + expires_delta
    return jwt.encode({"exp": expire, "sub": str(subject)}, get_secret_key(), algorithm=ALGORITHM)


def verify_password(plain: str, hashed: str) -> bool:
    """校验明文密码；空哈希（遗留假用户）一律视为失败，不可登录。"""
    if not hashed:
        return False
    ok, _ = password_hash.verify_and_update(plain, hashed)
    return ok


def get_password_hash(password: str) -> str:
    return password_hash.hash(password)
