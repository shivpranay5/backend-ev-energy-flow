from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

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


@dataclass(slots=True)
class ClerkSession:
    user_id: str
    email: str | None = None
    full_name: str | None = None


def _normalize_issuer(value: str | None) -> str | None:
    if not value:
        return None
    return value.strip().rstrip("/")


def _candidate_jwks_urls(settings: Settings, unverified_payload: dict[str, Any]) -> list[str]:
    candidates: list[str] = []
    configured = (settings.clerk_jwks_url or "").strip()
    if configured:
        candidates.append(configured)

    issuer = _normalize_issuer(settings.clerk_issuer) or _normalize_issuer(str(unverified_payload.get("iss") or ""))
    if issuer:
        candidates.append(f"{issuer}/.well-known/jwks.json")

    seen: set[str] = set()
    deduped: list[str] = []
    for url in candidates:
        if url and url not in seen:
            deduped.append(url)
            seen.add(url)
    return deduped


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


def _extract_email_from_payload(payload: dict[str, Any]) -> str | None:
    direct = payload.get("email") or payload.get("email_address")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    emails = payload.get("email_addresses")
    if isinstance(emails, list):
        for item in emails:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                value = item.get("email_address") or item.get("emailAddress")
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


async def verify_clerk_identity(request: Request, settings: Settings, bundle: DatabaseBundle) -> Principal:
    session = await verify_clerk_session(request, settings)
    if session.email:
        return await upsert_user_identity(
            bundle=bundle,
            user_id=session.user_id,
            email=session.email,
            full_name=session.full_name,
            source="clerk",
        )

    existing = await bundle.users.find_one({"clerk_user_id": session.user_id})
    if existing and existing.get("email"):
        return Principal(
            user_id=session.user_id,
            email=str(existing["email"]),
            role=str(existing.get("role") or "user"),
            full_name=existing.get("full_name"),
            source=str(existing.get("source") or "clerk"),
        )

    dev_email = request.headers.get("x-dev-email")
    if dev_email:
        return await _resolve_dev_identity(dev_email, bundle, source="dev-header")

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authenticated Clerk user has no synced email. Call /auth/sync after sign-in.",
    )


async def verify_clerk_session(request: Request, settings: Settings) -> ClerkSession:
    token = _extract_bearer_token(request)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing authentication token")

    try:
        unverified_payload = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_nbf": False,
                "verify_iat": False,
                "verify_aud": False,
                "verify_iss": False,
            },
            algorithms=["RS256", "EdDSA", "ES256", "HS256"],
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Malformed Clerk session token") from exc

    payload: dict[str, Any] | None = None
    last_exc: Exception | None = None
    for jwks_url in _candidate_jwks_urls(settings, unverified_payload):
        try:
            jwk_client = PyJWKClient(jwks_url)
            signing_key = await asyncio.to_thread(jwk_client.get_signing_key_from_jwt, token)
            payload = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "EdDSA", "ES256"],
                options={"verify_aud": False, "verify_iss": False},
            )
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc

    if payload is None:
        detail = f"Invalid Clerk session token: {last_exc}" if last_exc else "Invalid Clerk session token"
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)

    expected_issuer = _normalize_issuer(settings.clerk_issuer)
    actual_issuer = _normalize_issuer(str(payload.get("iss") or ""))
    if expected_issuer and actual_issuer != expected_issuer:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Clerk issuer: expected {expected_issuer}, got {actual_issuer or 'missing'}",
        )

    user_id = str(payload.get("sub") or "").strip()
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid Clerk token subject")

    full_name = payload.get("name")
    email = _extract_email_from_payload(payload)
    return ClerkSession(
        user_id=user_id,
        email=email,
        full_name=full_name if isinstance(full_name, str) else None,
    )


async def _resolve_dev_identity(email: str, bundle: DatabaseBundle, source: str) -> Principal:
    cleaned = email.strip().lower()
    user_id = f"dev_{cleaned}"
    return await upsert_user_identity(
        bundle=bundle,
        user_id=user_id,
        email=cleaned,
        full_name=None,
        source=source,
    )

async def upsert_user_identity(
    bundle: DatabaseBundle,
    user_id: str,
    email: str,
    full_name: str | None,
    source: str,
) -> Principal:
    cleaned = email.strip().lower()
    role = await _role_for_email(bundle, cleaned)
    now = _utc_now()
    await bundle.users.update_one(
        {"$or": [{"clerk_user_id": user_id}, {"email": cleaned}]},
        {
            "$set": {
                "clerk_user_id": user_id,
                "email": cleaned,
                "role": role,
                "full_name": full_name,
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
        full_name=full_name,
        source=source,
    )


async def _role_for_email(bundle: DatabaseBundle, email: str) -> str:
    admin = await bundle.admins.find_one({"email": email, "active": True})
    return "admin" if admin else "user"


def _utc_now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)
