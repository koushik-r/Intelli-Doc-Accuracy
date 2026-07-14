"""Extraction templates (a.k.a. schemas).

A "template" is the set of fields we want to pull out of a document. The same
template drives both the LLM path (rendered into the prompt) and the ACU path
(rendered into an analyzer field schema), so a user picks one template and both
extraction methods target the identical field set — which is what makes the
Phase 4 consensus comparison meaningful.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class FieldType(str, Enum):
    STRING = "string"
    NUMBER = "number"
    DATE = "date"
    BOOLEAN = "boolean"
    ARRAY = "array"


class FieldSpec(BaseModel):
    """One field to extract."""

    name: str
    description: str = ""
    type: FieldType = FieldType.STRING
    # Hint about where this field usually lives; used to bias the merge
    # (a table-sourced value beats a narrative-sourced one on ties).
    prefers_table: bool = False


class ExtractionTemplate(BaseModel):
    """A named collection of fields to extract from a document."""

    template_id: str
    name: str
    description: str = ""
    fields: list[FieldSpec] = Field(default_factory=list)

    def field_names(self) -> list[str]:
        return [f.name for f in self.fields]


# --- A couple of ready-made templates for the demo ------------------------

FNOL_TEMPLATE = ExtractionTemplate(
    template_id="fnol",
    name="First Notice of Loss (FNOL)",
    description="Insurance first-notice-of-loss claim intake form.",
    fields=[
        FieldSpec(name="policy_number", description="The insurance policy number", prefers_table=True),
        FieldSpec(name="claim_number", description="Assigned claim/reference number", prefers_table=True),
        FieldSpec(name="claimant_name", description="Full name of the claimant"),
        FieldSpec(name="date_of_loss", description="Date the loss/incident occurred", type=FieldType.DATE, prefers_table=True),
        FieldSpec(name="time_of_loss", description="Time the loss/incident occurred"),
        FieldSpec(name="loss_location", description="Address or location where the loss occurred"),
        FieldSpec(name="loss_description", description="Narrative description of what happened"),
        FieldSpec(name="estimated_amount", description="Estimated loss/damage amount", type=FieldType.NUMBER, prefers_table=True),
    ],
)

GENERIC_TEMPLATE = ExtractionTemplate(
    template_id="generic",
    name="Generic Document",
    description="Loose key fields for an arbitrary business document.",
    fields=[
        FieldSpec(name="document_type", description="What kind of document this is"),
        FieldSpec(name="document_date", description="Primary date on the document", type=FieldType.DATE),
        FieldSpec(name="parties", description="Named people or organizations", type=FieldType.ARRAY),
        FieldSpec(name="reference_number", description="Any primary reference/ID number", prefers_table=True),
        FieldSpec(name="total_amount", description="Any headline monetary total", type=FieldType.NUMBER, prefers_table=True),
        FieldSpec(name="summary", description="One-paragraph summary of the document"),
    ],
)

TEMPLATES: dict[str, ExtractionTemplate] = {
    FNOL_TEMPLATE.template_id: FNOL_TEMPLATE,
    GENERIC_TEMPLATE.template_id: GENERIC_TEMPLATE,
}


def get_template(template_id: str) -> ExtractionTemplate:
    if template_id not in TEMPLATES:
        raise KeyError(f"Unknown template '{template_id}'. Known: {list(TEMPLATES)}")
    return TEMPLATES[template_id]


def register_template(template: ExtractionTemplate) -> ExtractionTemplate:
    """Register a template (e.g. one generated from a document) so both
    extraction paths can be run against it by id."""
    TEMPLATES[template.template_id] = template
    return template
