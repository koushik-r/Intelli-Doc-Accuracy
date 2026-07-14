"""Template generation — infer an extraction template from a sample document.

When a user has no predefined template, they upload a document and we infer one:
run structured extraction, then propose an ExtractionTemplate (a field set) that
the user keeps and runs the LLM/ACU paths against.

Two engines:
  * generate_template_llm       — the model proposes reusable fields (Foundry).
  * generate_template_heuristic — parses Label:value pairs + 2-col table rows.
    No LLM required, so template generation still works offline.
"""

from __future__ import annotations

import re

from extraction.LLM.LLM import ModelClient, complete_json, extract_document_structured
from models.template import ExtractionTemplate, FieldSpec, FieldType
from models.usage import Usage

_LABEL_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 /&#().'-]{1,48}?)\s*[:#]\s*(.+?)\s*$")


def _slugify(label: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or "field"


def _guess_type(value: str) -> FieldType:
    v = value.strip()
    if re.search(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}", v):
        return FieldType.DATE
    if re.fullmatch(r"[$£€]?\s*[\d,]+(\.\d+)?%?", v):
        return FieldType.NUMBER
    if v.lower() in {"yes", "no", "true", "false"}:
        return FieldType.BOOLEAN
    return FieldType.STRING


def _collect_candidates(structured_pages: list[dict]) -> list[tuple[str, str, bool]]:
    candidates: list[tuple[str, str, bool]] = []
    seen: set[str] = set()

    def add(label: str, value: str, from_table: bool) -> None:
        slug = _slugify(label)
        if slug in seen or len(slug) < 2:
            return
        seen.add(slug)
        candidates.append((label.strip(), value.strip(), from_table))

    for page in structured_pages:
        for line in page["tables"].splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) == 2 and cells[0] and set(cells[0]) != {"-"}:
                if cells[0].lower() in {"field", "label", "key"}:
                    continue
                add(cells[0], cells[1], True)
        for raw_line in page["narrative"].splitlines():
            for segment in raw_line.split(" | "):
                m = _LABEL_RE.match(segment)
                if m:
                    add(m.group(1), m.group(2).lstrip(":#").strip(), False)
    return candidates


def generate_template_heuristic(
    structured_pages: list[dict], template_id: str, name: str
) -> ExtractionTemplate:
    fields = [
        FieldSpec(
            name=_slugify(label),
            description=f'"{label}" (e.g. "{sample[:40]}")',
            type=_guess_type(sample),
            prefers_table=from_table,
        )
        for label, sample, from_table in _collect_candidates(structured_pages)
    ]
    return ExtractionTemplate(
        template_id=template_id,
        name=name,
        description="Auto-generated template (heuristic) from a sample document.",
        fields=fields,
    )


_LLM_SYSTEM = """You are a document-schema designer. Given the structured content
of a sample document, propose the set of fields worth extracting from documents
of this kind. Return concise, reusable field definitions — not the specific
values in this one document.

Return JSON of shape:
{"document_type": "<short label>",
 "fields": [
   {"name": "<snake_case>", "description": "<what to extract>",
    "type": "string|number|date|boolean|array", "prefers_table": <bool>}
 ]}

Set prefers_table=true for fields that appear inside a grid/table region
(markdown tables in the input). Keep to the 5-20 most useful fields."""


def generate_template_llm(
    structured_pages: list[dict],
    client: ModelClient,
    template_id: str,
    name: str | None = None,
) -> tuple[ExtractionTemplate, Usage]:
    content = "\n\n".join(p["anchored"] for p in structured_pages[:5])
    obj, usage = complete_json(
        client, _LLM_SYSTEM, f"Structured document content:\n\n{content}", temperature=0.0
    )

    doc_type = (obj.get("document_type") or "document") if isinstance(obj, dict) else "document"
    raw_fields = obj.get("fields", []) if isinstance(obj, dict) else []
    fields: list[FieldSpec] = []
    for f in raw_fields:
        if not isinstance(f, dict) or not f.get("name"):
            continue
        try:
            ftype = FieldType(f.get("type", "string"))
        except ValueError:
            ftype = FieldType.STRING
        fields.append(
            FieldSpec(
                name=_slugify(f["name"]),
                description=str(f.get("description", "")),
                type=ftype,
                prefers_table=bool(f.get("prefers_table", False)),
            )
        )

    template = ExtractionTemplate(
        template_id=template_id,
        name=name or f"Auto: {doc_type}",
        description=f"Auto-generated template (LLM) for '{doc_type}'.",
        fields=fields,
    )
    return template, usage


def generate_template(
    pdf_bytes: bytes,
    template_id: str,
    name: str | None = None,
    client: ModelClient | None = None,
) -> tuple[ExtractionTemplate, Usage]:
    """Generate a template from a document. Uses the LLM engine if `client` is
    provided, else the offline heuristic. Returns (template, usage)."""
    structured_pages = extract_document_structured(pdf_bytes)
    if client is not None:
        return generate_template_llm(structured_pages, client, template_id, name)
    template = generate_template_heuristic(
        structured_pages, template_id, name or "Auto-generated template"
    )
    return template, Usage()
