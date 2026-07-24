import os
import jwt
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from bson import ObjectId
from app.database.database import db

JWT_SECRET = os.getenv("JWT_SECRET", "hitloop-jwt-secret-2026-production-key")
JWT_ALGORITHM = "HS256"
JWT_EXPIRY_HOURS = 72
JWT_REMEMBER_ME_EXPIRY_DAYS = 30

security = HTTPBearer(auto_error=False)


def create_access_token(user_id: str, remember_me: bool = False) -> str:
    if remember_me:
        expiry = datetime.utcnow() + timedelta(days=JWT_REMEMBER_ME_EXPIRY_DAYS)
    else:
        expiry = datetime.utcnow() + timedelta(hours=JWT_EXPIRY_HOURS)
    payload = {
        "sub": user_id,
        "exp": expiry,
        "iat": datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Session expired. Please log in again.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid authentication token.")


async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
    token = credentials.credentials
    payload = decode_token(token)
    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload.")
    try:
        user = await db["users"].find_one({"_id": ObjectId(user_id)})
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token payload.")
    if not user:
        raise HTTPException(status_code=401, detail="User not found. Session may have been revoked.")
    if user.get("account_status") == "disabled":
        raise HTTPException(status_code=403, detail="Account has been disabled.")
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[dict]:
    if not credentials or not credentials.credentials:
        return None
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        user_id = payload.get("sub")
        if user_id:
            user = await db["users"].find_one({"_id": ObjectId(user_id)})
            return user
    except Exception:
        return None
    return None


def require_role(*allowed_roles: str):
    async def _check(current_user: dict = Depends(get_current_user)) -> dict:
        role = current_user.get("role", "candidate")
        if role not in allowed_roles:
            raise HTTPException(status_code=403, detail=f"Access denied. Required role: {', '.join(allowed_roles)}.")
        return current_user
    return _check


def serialize_user(user: dict) -> dict:
    return {
        "id": str(user["_id"]),
        "full_name": user.get("full_name", ""),
        "email": user.get("email", ""),
        "phone": user.get("phone"),
        "role": user.get("role", "candidate"),
        "account_status": user.get("account_status", "active"),
        "email_verified": user.get("email_verified", False),
        "last_login": user.get("last_login"),
        "created_at": user.get("created_at", ""),
        "updated_at": user.get("updated_at", ""),
    }
