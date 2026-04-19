from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
import jwt
from fastapi import HTTPException, Request, status
from jwt import PyJWKClient

from .db import DatabaseBundle
from .settings import Settings


@dataclass(slots=True)
class Principal:
    user_id: str
    email: str
    role: str
    full_name: str | None = None
    source: str = "clerk"


def _extract_bearer_token(request: Request) -> str | None:
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    session_cookie = request.cookies.get("__session")
    if session_cookie:
        return session_cookie.strip()
    query_token = request.query_params.get("token")
    if query_token:
        return query_token.strip()
    return None


def _extract_email_from_clerk_user(user: dict[str, Any]) -> str | None:
    # Support both camelCase and snake_case payloads from Clerk responses.
    primary_id = user.get("primaryEmailAddressId") or user.get("primary_email_address_id")

    for key in ("emailAddresses", "email_addresses"):
        emails = user.get(key) or []
        if not isinstance(emails, list):
            continue
        if primary_id:
            for email in emails:
                if not isinstance(email, dict):
                    continue
                if email.get("id") == primary_id:
                    return email.get("emailAddress") or email.get("email_address")
        if emails:
            first = emails[0]
            if isinstance(first, dict):
                return first.get("emailAddress") or first.get("email_address")

    primary = user.get("primaryEmailAddress") or user.get("primary_email_address")
    if isinstance(primary, dict):
        return primary.get("emailAddress") or primary.get("email_address")

    return None


async def _fetch_clerk_user(settings: Settings, user_id: str) -> dict[str, Any]:
    if not settings.clerk_secret_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Clerk secret key is not configured on the backend",
        )

    url = f"https://api.clerk.com/v1/users/{user_id}"
    headers = {"Authorization": f"Bearer {settings.clerk_secret_key}"}
    async with httpx.AsyncClient(timeout=15) as client:
        response = await client.get(url, headers=headers)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Failed to fetch Clerk user profile",
        )
    return response.json()


async def verify_clerk_identity(request: Request, settings: Settings, bundle: DatabaseBundle) -> Principal:
    token = _extract_bearer_token(request)
    if not token:
        dev_email = request.headers.get("x-dev-email")
        if dev_email:
            return await _resolve_dev_identity(dev_email, bundle, source="dev-header")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    try:
        jwk_client = PyJWKClient(settings.clerk_jwks_url)
        signing_key = await asyncio.to_thread(jwk_client.get_signing_key_from_jwt, token)
        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            issuer=settings.clerk_issuer or None,
            options={"verify_aud": False},
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Clerk session token") from exc

    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Clerk token subject")

    clerk_user = await _fetch_clerk_user(settings, user_id)
    email = _extract_email_from_clerk_user(clerk_user)
    if not email:
        dev_email = request.headers.get("x-dev-email")
        if dev_email:
            email = dev_email.strip()
        else:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Clerk user has no email address")

    principal = await _resolve_email_identity(user_id, email, clerk_user, bundle, source="clerk")
    return principal


async def _resolve_dev_identity(email: str, bundle: DatabaseBundle, source: str) -> Principal:
    cleaned = email.strip().lower()
    role = await _role_for_email(bundle, cleaned)
    user_id = f"dev_{cleaned}"
    await bundle.users.update_one(
        {"email": cleaned},
        {
            "$set": {
                "email": cleaned,
                "role": role,
                "source": source,
                "updated_at": _utc_now(),
            },
            "$setOnInsert": {"created_at": _utc_now(), "clerk_user_id": user_id},
        },
        upsert=True,
    )
    return Principal(user_id=user_id, email=cleaned, role=role, source=source)


async def _resolve_email_identity(
    user_id: str,
    email: str,
    clerk_user: dict[str, Any],
    bundle: DatabaseBundle,
    source: str,
) -> Principal:
    cleaned = email.strip().lower()
    role = await _role_for_email(bundle, cleaned)
    now = _utc_now()
    await bundle.users.update_one(
        {"clerk_user_id": user_id},
        {
            "$set": {
                "clerk_user_id": user_id,
                "email": cleaned,
                "role": role,
                "full_name": clerk_user.get("fullName") or clerk_user.get("full_name"),
                "source": source,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        },
        upsert=True,
    )
    return Principal(
        user_id=user_id,
        email=cleaned,
        role=role,
        full_name=clerk_user.get("fullName") or clerk_user.get("full_name"),
        source=source,
    )


async def _role_for_email(bundle: DatabaseBundle, email: str) -> str:
    admin = await bundle.admins.find_one({"email": email, "active": True})
    return "admin" if admin else "user"


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)

