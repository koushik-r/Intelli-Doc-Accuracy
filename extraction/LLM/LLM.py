"""LLM path — structured layout extraction + field extraction via Azure AI Foundry.

One file for the whole LLM path:
  * Structured PDF layout (pdfplumber; PyMuPDF fallback) — tables vs narrative
    split with incremental confirmed bboxes, full-2D overlap, `|` column breaks.
  * FoundryClient — Azure AI Foundry (OpenAI-compatible chat completions).
  * Two prompts (strict table pass, permissive narrative pass) + chunking + a
    confidence-weighted, table-preferring merge.
"""

from __future__ import annotations

import io
import json
import logging
import os
from typing import Protocol

import pdfplumber

from models.extraction_result import ExtractedField, ExtractionResult, Method, SourceType
from models.template import ExtractionTemplate
from models.usage import Usage

logger = logging.getLogger(__name__)

# --- layout tunables ------------------------------------------------------- #
MIN_TABLE_ROWS = 2
MIN_TABLE_COLS = 2
WORD_OVERLAP_THRESHOLD = 0.5
LINE_Y_ROUND = 3
COLUMN_GAP_PT = 15
NARRATIVE_STRIP_WARN = 0.95
SCANNED_CHAR_FLOOR = 10
SCANNED_DENSITY_THRESHOLD = 1 / 5000

DEFAULT_CHUNK_TOKENS = 6000


# ==========================================================================
# Geometry helpers
# ==========================================================================
def _rect_intersection_area(a: tuple, b: tuple) -> float:
    ax0, atop, ax1, abot = a
    bx0, btop, bx1, bbot = b
    ix0, iy0 = max(ax0, bx0), max(atop, btop)
    ix1, iy1 = min(ax1, bx1), min(abot, bbot)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def _word_rect(word: dict) -> tuple:
    return (word["x0"], word["top"], word["x1"], word["bottom"])


def _word_in_bbox(word: dict, bbox: tuple, threshold: float = WORD_OVERLAP_THRESHOLD) -> bool:
    """True if >threshold of the WORD's own area lies inside bbox (full 2D)."""
    wr = _word_rect(word)
    warea = (wr[2] - wr[0]) * (wr[3] - wr[1])
    if warea <= 0:
        return False
    return (_rect_intersection_area(wr, bbox) / warea) > threshold


def _in_any_bbox(word: dict, bboxes: list[tuple], threshold: float = WORD_OVERLAP_THRESHOLD) -> bool:
    return any(_word_in_bbox(word, b, threshold) for b in bboxes)


# ==========================================================================
# Table plausibility + markdown
# ==========================================================================
def _is_plausible_table(rows: list[list]) -> bool:
    if not rows or len(rows) < MIN_TABLE_ROWS:
        return False
    n_cols = max((len(r) for r in rows), default=0)
    if n_cols < MIN_TABLE_COLS:
        return False
    non_empty_cols = 0
    for c in range(n_cols):
        if any(c < len(r) and (r[c] or "").strip() for r in rows):
            non_empty_cols += 1
    return non_empty_cols >= MIN_TABLE_COLS


def _table_to_markdown(rows: list[list]) -> str:
    cleaned = [[(cell or "").strip().replace("\n", " ") for cell in row] for row in rows]
    n_cols = max(len(r) for r in cleaned)
    cleaned = [r + [""] * (n_cols - len(r)) for r in cleaned]
    header, *body = cleaned
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * n_cols) + " |"]
    for row in body:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


# ==========================================================================
# Narrative reconstruction
# ==========================================================================
def _join_line(row: list[dict]) -> str:
    parts = [row[0]["text"]]
    for prev, cur in zip(row, row[1:]):
        gap = cur["x0"] - prev["x1"]
        parts.append(" | " if gap > COLUMN_GAP_PT else " ")
        parts.append(cur["text"])
    return "".join(parts)


def _words_to_narrative(words: list[dict]) -> str:
    if not words:
        return ""
    lines: dict[int, list[dict]] = {}
    for w in words:
        lines.setdefault(round(w["top"] / LINE_Y_ROUND), []).append(w)
    out_lines = []
    for key in sorted(lines):
        out_lines.append(_join_line(sorted(lines[key], key=lambda w: w["x0"])))
    return "\n".join(out_lines)


# ==========================================================================
# Page-level structured extraction
# ==========================================================================
def extract_structured_page(page) -> dict:
    confirmed_bboxes: list[tuple] = []
    table_markdowns: list[str] = []

    for tbl in page.find_tables():
        try:
            rows = tbl.extract()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("table.extract() failed on a candidate: %s", exc)
            continue
        if _is_plausible_table(rows):
            confirmed_bboxes.append(tbl.bbox)  # incremental — only if plausible
            table_markdowns.append(_table_to_markdown(rows))

    all_words = page.extract_words()
    kept = [w for w in all_words if not _in_any_bbox(w, confirmed_bboxes)]

    if (
        all_words
        and not confirmed_bboxes
        and (len(all_words) - len(kept)) / len(all_words) > NARRATIVE_STRIP_WARN
    ):  # pragma: no cover
        logger.warning("Stripped >%.0f%% of words with 0 tables; falling back to narrative.",
                       NARRATIVE_STRIP_WARN * 100)
        kept = all_words

    return {
        "tables": "\n\n".join(table_markdowns),
        "narrative": _words_to_narrative(kept),
        "table_count": len(confirmed_bboxes),
    }


def classify_page_type(page) -> str:
    """digital vs scanned by text density. TODO(ocr): route scanned pages to a
    vision/OCR fallback in a later phase."""
    chars = page.chars or []
    if len(chars) < SCANNED_CHAR_FLOOR:
        return "scanned"
    area = float(page.width) * float(page.height)
    if area <= 0:
        return "scanned"
    return "digital" if (len(chars) / area) > SCANNED_DENSITY_THRESHOLD else "scanned"


def extract_document_structured(pdf_bytes: bytes) -> list[dict]:
    """Structured extraction across all pages with <!-- PAGE N --> anchors."""
    pages: list[dict] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for idx, page in enumerate(pdf.pages, start=1):
            page_type = classify_page_type(page)
            if page_type == "scanned":
                # TODO(ocr): vision/OCR fallback. For now emit an empty page.
                struct = {"tables": "", "narrative": "", "table_count": 0}
            else:
                struct = extract_structured_page(page)

            body = "\n\n".join(x for x in (struct["tables"], struct["narrative"]) if x)
            anchored = f"<!-- PAGE {idx} -->\n{body}\n<!-- END PAGE {idx} -->"
            pages.append(
                {
                    "page": idx,
                    "page_type": page_type,
                    "tables": struct["tables"],
                    "narrative": struct["narrative"],
                    "table_count": struct["table_count"],
                    "anchored": anchored,
                }
            )
    return pages


# ==========================================================================
# Azure AI Foundry model client
# ==========================================================================
class ModelClient(Protocol):
    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 4096
    ) -> tuple[str, Usage]:
        ...


class ModelClientError(RuntimeError):
    """No usable Foundry provider is configured."""


class FoundryClient:
    """Azure AI Foundry chat-completions client (OpenAI-compatible surface).

    Works with a Foundry inference endpoint such as
    https://<resource>.services.ai.azure.com/models — the deployment/model name
    is sent in the body, auth via the `api-key` header.
    """

    def __init__(self, endpoint: str, api_key: str, deployment: str, api_version: str):
        import httpx

        base = endpoint.rstrip("/")
        self._url = f"{base}/chat/completions?api-version={api_version}"
        self._headers = {"api-key": api_key, "content-type": "application/json"}
        self._deployment = deployment
        self._httpx = httpx

    def complete(
        self, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 4096
    ) -> tuple[str, Usage]:
        payload = {
            "model": self._deployment,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = self._httpx.post(self._url, headers=self._headers, json=payload, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"]
        u = data.get("usage", {})
        return text, Usage.from_tokens(u.get("prompt_tokens", 0), u.get("completion_tokens", 0))


def get_model_client() -> ModelClient:
    """Build a Foundry client from environment variables."""
    endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
    key = os.getenv("AZURE_AI_FOUNDRY_KEY")
    if not endpoint or not key:
        raise ModelClientError(
            "Azure AI Foundry not configured. Set AZURE_AI_FOUNDRY_ENDPOINT and "
            "AZURE_AI_FOUNDRY_KEY in .env."
        )
    return FoundryClient(
        endpoint=endpoint,
        api_key=key,
        deployment=os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", "gpt-4o"),
        api_version=os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-05-01-preview"),
    )


def complete_json(
    client: ModelClient, system: str, user: str, *, temperature: float = 0.0, max_tokens: int = 4096
) -> tuple[object, Usage]:
    """Call the model and parse its reply as JSON (tolerates code fences/prose)."""
    text, usage = client.complete(
        system=system + "\n\nRespond with ONLY valid JSON. No prose, no code fences.",
        user=user,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return _parse_json_loose(text), usage


def _parse_json_loose(text: str) -> object:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = min((i for i in (text.find("{"), text.find("[")) if i != -1), default=-1)
        end = max(text.rfind("}"), text.rfind("]"))
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


# ==========================================================================
# Prompts
# ==========================================================================
_CONVENTIONS = """You are extracting fields from a document that has been
pre-structured by a layout engine. Read it using these exact conventions:

- A GitHub-style markdown table represents a real table/grid region of the page.
  Each row is a strict label-value or record pair — do not reinterpret it.
- Inside a plain text (non-table) line, a ` | ` marks a COLUMN BREAK. The text on
  either side belongs to DISTINCT fields. NEVER merge across a ` | `.
  Example: "Date of Occurrence: 11/18/2025 | Time: 2:45 PM" is TWO fields.
- `<!-- PAGE N -->` / `<!-- END PAGE N -->` are page-attribution markers, not
  document content. Use them to report which page each value came from.
- Only extract values actually present. If a field is absent, return null for it."""

_OUTPUT_SPEC = """Return JSON of this exact shape:
{"fields": {
   "<field_name>": {"value": <value|null>, "confidence": <0.0-1.0>,
                    "source_pages": [<int>, ...]}
}}
Include every requested field. confidence reflects how certain you are the value
is correct and complete."""


def _field_list(template: ExtractionTemplate) -> str:
    return "\n".join(
        f"- {f.name} ({f.type.value}): {f.description or f.name}" for f in template.fields
    )


def _table_system_prompt(template: ExtractionTemplate) -> str:
    return (
        _CONVENTIONS
        + "\n\nYou are working ONLY with the TABLE/grid regions. Match values"
        " exactly as written — do not normalize, infer, or paraphrase.\n\n"
        "Fields to extract:\n" + _field_list(template) + "\n\n" + _OUTPUT_SPEC
    )


def _narrative_system_prompt(template: ExtractionTemplate) -> str:
    return (
        _CONVENTIONS
        + "\n\nYou are working ONLY with the NARRATIVE (prose) regions. Extract"
        " the requested entities semantically. Respect ` | ` column breaks as"
        " field boundaries.\n\nFields to extract:\n"
        + _field_list(template) + "\n\n" + _OUTPUT_SPEC
    )


# ==========================================================================
# Chunking + extraction
# ==========================================================================
def _est_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def chunk_structured_pages(
    structured_pages: list[dict], max_tokens: int = DEFAULT_CHUNK_TOKENS
) -> list[list[dict]]:
    """Group whole pages into chunks under the token budget (pages never split)."""
    chunks: list[list[dict]] = []
    current: list[dict] = []
    current_tokens = 0
    for page in structured_pages:
        ptoks = _est_tokens(page["anchored"])
        if current and current_tokens + ptoks > max_tokens:
            chunks.append(current)
            current, current_tokens = [], 0
        current.append(page)
        current_tokens += ptoks
    if current:
        chunks.append(current)
    return chunks


def _content_for(pages: list[dict], key: str) -> str:
    parts = []
    for p in pages:
        if p[key].strip():
            parts.append(f"<!-- PAGE {p['page']} -->\n{p[key]}\n<!-- END PAGE {p['page']} -->")
    return "\n\n".join(parts)


def _parse_fields(obj: object, source_type: SourceType) -> dict[str, ExtractedField]:
    fields: dict[str, ExtractedField] = {}
    if not isinstance(obj, dict):
        return fields
    raw = obj.get("fields", {})
    if not isinstance(raw, dict):
        return fields
    for name, spec in raw.items():
        if not isinstance(spec, dict):
            continue
        value = spec.get("value")
        if value is None or value == "":
            continue
        pages = spec.get("source_pages") or []
        fields[name] = ExtractedField(
            name=name,
            value=value,
            confidence=float(spec.get("confidence", 0.5) or 0.0),
            source_type=source_type,
            source_pages=[int(p) for p in pages if isinstance(p, (int, float))],
            method=Method.LLM,
        )
    return fields


def _better(a: ExtractedField, b: ExtractedField) -> ExtractedField:
    if abs(a.confidence - b.confidence) > 0.05:
        return a if a.confidence > b.confidence else b
    a_table = a.source_type == SourceType.TABLE
    b_table = b.source_type == SourceType.TABLE
    if a_table != b_table:
        return a if a_table else b
    return a if a.confidence >= b.confidence else b


def merge_chunk_results(
    chunk_results: list[dict[str, ExtractedField]],
) -> dict[str, ExtractedField]:
    merged: dict[str, ExtractedField] = {}
    for result in chunk_results:
        for name, field in result.items():
            merged[name] = field if name not in merged else _better(merged[name], field)
    return merged


def extract_fields_llm(
    structured_pages: list[dict],
    template: ExtractionTemplate,
    model_client: ModelClient,
    max_tokens_per_chunk: int = DEFAULT_CHUNK_TOKENS,
) -> ExtractionResult:
    """Run the full LLM extraction path and return a uniform ExtractionResult."""
    table_sys = _table_system_prompt(template)
    narr_sys = _narrative_system_prompt(template)

    per_pass_results: list[dict[str, ExtractedField]] = []
    total_usage = Usage()

    for chunk in chunk_structured_pages(structured_pages, max_tokens_per_chunk):
        table_content = _content_for(chunk, "tables")
        narr_content = _content_for(chunk, "narrative")

        if table_content:
            obj, usage = complete_json(model_client, table_sys, table_content, temperature=0.0)
            per_pass_results.append(_parse_fields(obj, SourceType.TABLE))
            total_usage = total_usage.add(usage)

        if narr_content:
            obj, usage = complete_json(model_client, narr_sys, narr_content, temperature=0.2)
            per_pass_results.append(_parse_fields(obj, SourceType.NARRATIVE))
            total_usage = total_usage.add(usage)

    merged = merge_chunk_results(per_pass_results)
    for name in template.field_names():
        merged.setdefault(name, ExtractedField(name=name, value=None, confidence=0.0, method=Method.LLM))

    return ExtractionResult(
        method=Method.LLM, template_id=template.template_id, fields=merged, usage=total_usage
    )
