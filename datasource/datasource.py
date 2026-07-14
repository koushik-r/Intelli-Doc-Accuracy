"""MongoDB connection + collection accessors.

A single cached client is reused across Azure Functions invocations (the module
stays warm between calls on the same worker). Connection string and database
name come from environment / app settings:

  MONGODB_URI   — e.g. mongodb://localhost:27017 or an Atlas SRV string
  MONGODB_DB    — database name (default: intellidoc)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from pymongo import MongoClient, ASCENDING
from pymongo.database import Database

DEFAULT_DB_NAME = "intellidoc"


@lru_cache(maxsize=1)
def _client() -> MongoClient:
    uri = os.getenv("MONGODB_URI")
    if not uri:
        raise RuntimeError(
            "MONGODB_URI is not set. Configure it in local.settings.json / app settings."
        )
    client: MongoClient = MongoClient(uri, tz_aware=True, appname="intellidoc-extraction")
    return client


def get_db() -> Database:
    db = _client()[os.getenv("MONGODB_DB", DEFAULT_DB_NAME)]
    return db


@dataclass(frozen=True)
class Collections:
    documents: str = "documents"
    templates: str = "templates"
    extraction_results: str = "extraction_results"
    usage: str = "usage"


collections = Collections()

_indexes_ready = False


def ensure_indexes() -> None:
    """Create the handful of indexes the IDP views rely on. Idempotent."""
    global _indexes_ready
    if _indexes_ready:
        return
    db = get_db()
    db[collections.documents].create_index([("created_at", ASCENDING)])
    db[collections.templates].create_index([("template_id", ASCENDING)], unique=True)
    db[collections.extraction_results].create_index([("document_id", ASCENDING)])
    db[collections.usage].create_index([("document_id", ASCENDING)])
    _indexes_ready = True
