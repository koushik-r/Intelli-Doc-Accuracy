"""Extraction result models.

`ExtractedField` carries not just a value but *where it came from* (which page,
whether it was a table or narrative region, which method) and how confident the
extractor is. That provenance is what powers the consensus/accuracy story: on a
disagreement we can explain *why* we trust one value over another instead of
picking blindly.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from .usage import Usage


class SourceType(str, Enum):
    TABLE = "table"
    NARRATIVE = "narrative"
    UNKNOWN = "unknown"


class Method(str, Enum):
    LLM = "llm"
    ACU = "acu"
    CONSENSUS = "consensus"


class ExtractedField(BaseModel):
    name: str
    value: object | None = None
    confidence: float = 0.0
    source_type: SourceType = SourceType.UNKNOWN
    source_pages: list[int] = Field(default_factory=list)
    method: Method = Method.LLM
    # Set by the consensus path when the two methods disagreed on this field.
    needs_review: bool = False


class FieldConflict(BaseModel):
    """Records a disagreement between the two extraction paths for one field."""

    name: str
    llm_value: object | None = None
    llm_confidence: float = 0.0
    acu_value: object | None = None
    acu_confidence: float = 0.0
    chosen: object | None = None
    chosen_method: Method | None = None


class ExtractionResult(BaseModel):
    """Uniform result shape returned by every extraction path."""

    method: Method
    template_id: str
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    usage: Usage = Field(default_factory=Usage)

    # Populated only by the consensus path.
    conflicts: list[FieldConflict] = Field(default_factory=list)
    agreement_rate: float | None = None

    # Free-form notes/warnings surfaced to the demo UI.
    notes: list[str] = Field(default_factory=list)

    def value_map(self) -> dict[str, object | None]:
        return {name: f.value for name, f in self.fields.items()}
