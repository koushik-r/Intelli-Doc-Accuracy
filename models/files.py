"""MongoDB persistence models.

These are the shapes we actually store in Mongo. They wrap / flatten the pure
extraction models (`ExtractionResult`, `ExtractionTemplate`, `Usage`) into
documents with an `_id`, timestamps and cross-references so the IDP frontend can
list, preview and drill into everything that was produced for one upload.

Every model carries an `id` (mapped to Mongo's `_id`) plus `to_mongo()` /
`from_mongo()` helpers so the API layer never touches raw dicts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .extraction_result import ExtractionResult
from .template import ExtractionTemplate
from .usage import Usage


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    EXTRACTED = "extracted"
    FAILED = "failed"


class _MongoBase(BaseModel):
    """Base with id<->_id mapping and mongo (de)serialisation helpers."""

    id: str | None = Field(default=None, alias="_id")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    model_config = {"populate_by_name": True}

    def to_mongo(self) -> dict[str, Any]:
        data = self.model_dump(by_alias=True, exclude_none=False)
        # Let Mongo assign the _id when we don't have one yet.
        if data.get("_id") is None:
            data.pop("_id", None)
        return data

    @classmethod
    def from_mongo(cls, doc: dict[str, Any] | None):
        if doc is None:
            return None
        doc = dict(doc)
        if "_id" in doc:
            doc["_id"] = str(doc["_id"])
        return cls.model_validate(doc)


class DocumentRecord(_MongoBase):
    """One uploaded document, stored in Azure Blob and tracked in Mongo."""

    original_filename: str
    content_type: str = "application/octet-stream"
    size_bytes: int = 0

    # Azure Blob provenance
    blob_container: str = ""
    blob_name: str = ""
    blob_url: str = ""

    status: DocumentStatus = DocumentStatus.UPLOADED
    template_id: str | None = None

    # Denormalised pointers filled in once extraction completes.
    extraction_result_id: str | None = None
    overall_confidence: float | None = None
    error: str | None = None


class TemplateRecord(_MongoBase):
    """A stored extraction template (inferred from a sample doc or predefined)."""

    template_id: str
    name: str
    description: str = ""
    source: str = "manual"  # "manual" | "inferred" | "builtin"
    # The full template definition (field specs) as produced by the library.
    definition: ExtractionTemplate
    # Optional pointer back to the sample document it was inferred from.
    sample_document_id: str | None = None


class ExtractionResultRecord(_MongoBase):
    """Persisted extraction output + confidence per field."""

    document_id: str
    template_id: str
    method: str
    overall_confidence: float = 0.0
    # The full uniform result from the library (fields, conflicts, usage...).
    result: ExtractionResult


class UsageRecord(_MongoBase):
    """Token / cost usage for one extraction run, kept separately so the IDP
    'usage' view can aggregate cheaply without loading full results."""

    document_id: str
    extraction_result_id: str | None = None
    template_id: str | None = None
    method: str = "llm"
    usage: Usage = Field(default_factory=Usage)


__all__ = [
    "DocumentStatus",
    "DocumentRecord",
    "TemplateRecord",
    "ExtractionResultRecord",
    "UsageRecord",
]
