from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _split_csv(value: str | None, default: list[str]) -> list[str]:
    if not value:
        return default
    parts = [item.strip() for item in value.split(",")]
    return [item for item in parts if item]


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str = field(default_factory=lambda: os.getenv("MONGODB_URI", "").strip())
    mongodb_db: str = field(default_factory=lambda: os.getenv("MONGODB_DB", "openvpp").strip() or "openvpp")
    clerk_jwks_url: str = field(
        default_factory=lambda: os.getenv("CLERK_JWKS_URL", "https://api.clerk.com/v1/jwks").strip()
    )
    clerk_secret_key: str = field(default_factory=lambda: os.getenv("CLERK_SECRET_KEY", "").strip())
    clerk_issuer: str = field(default_factory=lambda: os.getenv("CLERK_ISSUER", "").strip())
    cors_origins: list[str] = field(
        default_factory=lambda: _split_csv(
            os.getenv("CORS_ORIGINS"),
            ["http://localhost:5173", "http://127.0.0.1:5173"],
        )
    )
    cors_origin_regex: str = field(
        default_factory=lambda: os.getenv(
            "CORS_ORIGIN_REGEX",
            r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
        ).strip()
    )
    allow_guest_ws: bool = field(default_factory=lambda: os.getenv("ALLOW_GUEST_WS", "true").lower() == "true")
    admin_emails: tuple[str, ...] = (
        "orian.neo007@gmail.com",
        "ppranayreddy5454@gmail.com",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
