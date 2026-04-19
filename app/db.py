from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from .settings import Settings


@dataclass(slots=True)
class DatabaseBundle:
    client: AsyncIOMotorClient
    db: AsyncIOMotorDatabase

    @property
    def admins(self):
        return self.db["admins"]

    @property
    def users(self):
        return self.db["users"]

    @property
    def simulation_runs(self):
        return self.db["simulation_runs"]


async def connect_mongo(settings: Settings) -> DatabaseBundle:
    if not settings.mongodb_uri:
        raise RuntimeError("MONGODB_URI is required")
    client = AsyncIOMotorClient(settings.mongodb_uri)
    db = client[settings.mongodb_db]
    bundle = DatabaseBundle(client=client, db=db)
    await ensure_indexes(bundle)
    return bundle


async def ensure_indexes(bundle: DatabaseBundle) -> None:
    await bundle.admins.create_index("email", unique=True)
    await bundle.users.create_index("email", unique=True, sparse=True)
    await bundle.users.create_index("clerk_user_id", unique=True, sparse=True)
    await bundle.simulation_runs.create_index("created_at")


async def seed_admins(bundle: DatabaseBundle, admin_emails: tuple[str, ...]) -> list[dict[str, Any]]:
    now = datetime.now(timezone.utc)
    seeded: list[dict[str, Any]] = []
    for email in admin_emails:
        doc = {
            "email": email,
            "role": "admin",
            "active": True,
            "seeded": True,
            "updated_at": now,
            "created_at": now,
        }
        await bundle.admins.update_one(
            {"email": email},
            {
                "$set": {
                    "role": "admin",
                    "active": True,
                    "seeded": True,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                },
            },
            upsert=True,
        )
        seeded.append(doc)
    return seeded

