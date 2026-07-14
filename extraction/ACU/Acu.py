"""Azure Content Understanding (ACU) path — client, overlap batching, merge.

One file for the whole ACU path:
  * ACUClient          — REST client; failures raise ACUAnalysisError (never None).
  * page/size pre-check + overlapping batch splitting for >150-page / >100MB docs
  * per-batch context injection + failure-tolerant run
  * merge_acu_batches  — overlap-zone dedup, core-over-overlap tie-breaking
  * template <-> analyzer schema + raw-result normalization helpers

The client is injected everywhere, so a mock and the real client are
interchangeable and the batching logic never imports a concrete client.
"""

from __future__ import annotations

import logging
import time

import fitz  # PyMuPDF
import httpx

from models.template import ExtractionTemplate, FieldType

logger = logging.getLogger(__name__)

DEFAULT_PAGE_LIMIT = 150
DEFAULT_SIZE_LIMIT_MB = 100
DEFAULT_OVERLAP = 5


# ==========================================================================
# Client
# ==========================================================================
class ACUError(RuntimeError):
    """Base class for ACU client errors."""


class ACUAnalysisError(ACUError):
    def __init__(self, message: str, status_code: int | None = None, body: object = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        return f"{super().__str__()} (status={self.status_code}, body={self.body!r})"


class ACUClient:
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        api_version: str = "2024-12-01-preview",
        poll_interval_s: float = 2.0,
        timeout_s: float = 300.0,
    ):
        self._endpoint = endpoint.rstrip("/")
        self._api_version = api_version
        self._headers = {"Ocp-Apim-Subscription-Key": api_key}
        self._poll_interval_s = poll_interval_s
        self._timeout_s = timeout_s

    def create_or_replace_analyzer(self, analyzer_id: str, schema: dict) -> dict:
        url = (
            f"{self._endpoint}/contentunderstanding/analyzers/"
            f"{analyzer_id}?api-version={self._api_version}"
        )
        resp = httpx.put(url, headers=self._headers, json=schema, timeout=self._timeout_s)
        if resp.status_code >= 400:
            raise ACUAnalysisError(
                "Failed to create/replace analyzer", resp.status_code, _safe_body(resp)
            )
        return _safe_body(resp)

    def analyze_document(
        self, analyzer_id: str, document_bytes: bytes, content_type: str = "application/pdf"
    ) -> dict:
        """POST to analyzeBinary, then poll Operation-Location until terminal.

        Raises ACUAnalysisError on any non-success — never returns None.
        """
        url = (
            f"{self._endpoint}/contentunderstanding/analyzers/{analyzer_id}:analyzeBinary"
            f"?api-version={self._api_version}"
        )
        headers = {**self._headers, "Content-Type": content_type}
        resp = httpx.post(url, headers=headers, content=document_bytes, timeout=self._timeout_s)
        if resp.status_code not in (200, 201, 202):
            raise ACUAnalysisError(
                "analyzeBinary submission failed", resp.status_code, _safe_body(resp)
            )

        op_location = resp.headers.get("Operation-Location")
        if not op_location:
            if resp.status_code == 200:
                return _safe_body(resp)
            raise ACUAnalysisError(
                "No Operation-Location header on accepted request", resp.status_code, _safe_body(resp)
            )
        return self._poll(op_location)

    def _poll(self, op_location: str) -> dict:
        deadline = self._timeout_s
        waited = 0.0
        while True:
            resp = httpx.get(op_location, headers=self._headers, timeout=self._timeout_s)
            body = _safe_body(resp)
            if resp.status_code >= 400:
                raise ACUAnalysisError("Polling failed", resp.status_code, body)
            status = (body.get("status") or "").lower() if isinstance(body, dict) else ""
            if status in ("succeeded", "completed"):
                return body
            if status in ("failed", "cancelled"):
                raise ACUAnalysisError(f"Analysis {status}", resp.status_code, body)
            if waited >= deadline:
                raise ACUAnalysisError("Analysis timed out", None, body)
            time.sleep(self._poll_interval_s)
            waited += self._poll_interval_s


def _safe_body(resp: httpx.Response) -> object:
    try:
        return resp.json()
    except Exception:
        return resp.text


# ==========================================================================
# Page pre-check + overlap batching
# ==========================================================================
def get_page_count_and_size(pdf_bytes: bytes) -> tuple[int, float]:
    """(page_count, size_mb) — cheap, local, no ACU call."""
    size_mb = len(pdf_bytes) / (1024 * 1024)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        return doc.page_count, size_mb


def split_pdf_into_batches(
    pdf_bytes: bytes, max_pages: int, overlap_pages: int = DEFAULT_OVERLAP
) -> list[dict]:
    """Split into overlapping batches.

    Each batch: {batch_id, page_start, page_end, core_end, bytes} with 1-based,
    inclusive page numbers. `core_end` = end of the trusted (non-overlap) region.
    Consecutive batches advance by (max_pages - overlap_pages) so their cores are
    contiguous and every page lands in exactly one core.
    """
    if max_pages <= overlap_pages:
        raise ValueError("max_pages must be greater than overlap_pages")

    with fitz.open(stream=pdf_bytes, filetype="pdf") as src:
        total = src.page_count
        step = max_pages - overlap_pages
        batches: list[dict] = []
        batch_id = 0
        start = 1
        while start <= total:
            end = min(start + max_pages - 1, total)
            core_end = min(end, start + step - 1) if end < total else end

            out = fitz.open()
            out.insert_pdf(src, from_page=start - 1, to_page=end - 1)
            batch_bytes = out.tobytes()
            out.close()

            batches.append(
                {
                    "batch_id": batch_id,
                    "page_start": start,
                    "page_end": end,
                    "core_end": core_end,
                    "bytes": batch_bytes,
                }
            )
            if end >= total:
                break
            batch_id += 1
            start += step
        return batches


def extract_document_context(pdf_bytes: bytes, max_chars: int = 2000) -> str:
    """Cheap page-1 text used as injected 'document identity' context, so a batch
    starting on page 400 still knows the policy/claimant/doc type."""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        if doc.page_count == 0:
            return ""
        text = doc.load_page(0).get_text("text") or ""
    return text.strip()[:max_chars]


def run_acu_with_batching(
    pdf_bytes: bytes,
    analyzer_id: str,
    content_type: str,
    acu_client,
    page_limit: int = DEFAULT_PAGE_LIMIT,
    size_limit_mb: float = DEFAULT_SIZE_LIMIT_MB,
    overlap_pages: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """Analyze a document, batching only if it exceeds a limit.

    A single under-limit document returns a one-item list. Individual batch
    failures are logged/recorded but do NOT abort the remaining batches.
    """
    page_count, size_mb = get_page_count_and_size(pdf_bytes)
    context = extract_document_context(pdf_bytes)

    if page_count <= page_limit and size_mb <= size_limit_mb:
        result = acu_client.analyze_document(analyzer_id, pdf_bytes, content_type)
        return [
            {
                "batch_id": 0,
                "page_start": 1,
                "page_end": page_count,
                "core_end": page_count,
                "context": context,
                "analysis_result": result,
                "error": None,
            }
        ]

    logger.info(
        "Document exceeds ACU limits (pages=%d, size=%.1fMB) -> batching.", page_count, size_mb
    )
    batches = split_pdf_into_batches(pdf_bytes, page_limit, overlap_pages)
    results: list[dict] = []
    for b in batches:
        entry = {k: b[k] for k in ("batch_id", "page_start", "page_end", "core_end")}
        entry["context"] = context
        try:
            entry["analysis_result"] = acu_client.analyze_document(
                analyzer_id, b["bytes"], content_type
            )
            entry["error"] = None
        except Exception as exc:
            logger.warning("Batch %d failed: %s", b["batch_id"], exc)
            entry["analysis_result"] = None
            entry["error"] = str(exc)
        results.append(entry)
    return results


# ==========================================================================
# Merge
# ==========================================================================
def _batch_local_to_global(batch: dict, local_page: int) -> int:
    return batch["page_start"] + (local_page - 1)


def merge_acu_batches(batch_results: list[dict]) -> dict:
    """Merge per-batch ACU results into a single result-shaped dict.

    Overlap-zone duplicates: identical value -> keep one; disagreement -> keep
    the value from whichever batch holds that page in its CORE range (content
    near a hard boundary is more likely a fragment). Usage is summed; markdown
    is concatenated with page-anchor separators.
    """
    chosen: dict[str, dict] = {}
    markdown_parts: list[str] = []
    input_tokens = output_tokens = 0
    batch_count = 0
    errors: list[str] = []

    for entry in batch_results:
        result = entry.get("analysis_result")
        if entry.get("error"):
            errors.append(f"batch {entry['batch_id']}: {entry['error']}")
        if not result:
            continue
        batch_count += 1

        usage = result.get("usage") or {}
        input_tokens += int(usage.get("input_tokens", 0))
        output_tokens += int(usage.get("output_tokens", 0))

        md = result.get("markdown")
        if md:
            markdown_parts.append(
                f"<!-- BATCH {entry['batch_id']} PAGES "
                f"{entry['page_start']}-{entry['page_end']} -->\n{md}"
            )

        for name, spec in (result.get("fields") or {}).items():
            if not isinstance(spec, dict):
                continue
            local_page = int(spec.get("source_page", 1))
            global_page = _batch_local_to_global(entry, local_page)
            candidate = {
                "value": spec.get("value"),
                "confidence": float(spec.get("confidence", 0.5) or 0.0),
                "page": global_page,
                "batch_id": entry["batch_id"],
                "in_overlap_zone": global_page > entry["core_end"],
            }
            _merge_field(chosen, name, candidate)

    return {
        "fields": chosen,
        "markdown": "\n\n".join(markdown_parts),
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "batch_count": batch_count,
        "errors": errors,
    }


def _merge_field(chosen: dict[str, dict], name: str, cand: dict) -> None:
    prev = chosen.get(name)
    if prev is None:
        chosen[name] = cand
        return
    if prev["value"] == cand["value"]:
        if prev["in_overlap_zone"] and not cand["in_overlap_zone"]:
            chosen[name] = cand
        return
    prev_core = not prev["in_overlap_zone"]
    cand_core = not cand["in_overlap_zone"]
    if cand_core and not prev_core:
        chosen[name] = cand
    elif prev_core and not cand_core:
        return
    else:
        if cand["confidence"] > prev["confidence"]:
            chosen[name] = cand


# ==========================================================================
# Template <-> analyzer schema, and raw-result normalization
# ==========================================================================
_ACU_TYPE = {
    FieldType.STRING: "string",
    FieldType.NUMBER: "number",
    FieldType.DATE: "date",
    FieldType.BOOLEAN: "boolean",
    FieldType.ARRAY: "array",
}


def template_to_analyzer_schema(template: ExtractionTemplate) -> dict:
    """Render a template into a Content Understanding analyzer field schema."""
    fields = {
        f.name: {"type": _ACU_TYPE.get(f.type, "string"), "description": f.description or f.name}
        for f in template.fields
    }
    return {
        "description": template.description or template.name,
        "scenario": "document",
        "fieldSchema": {"fields": fields},
    }


def acu_raw_to_fields(analysis_result: dict) -> dict:
    """Normalize a raw ACU analysis result into
    {name: {value, confidence, source_page}} for merge_acu_batches. Defensive:
    ACU's shape varies by API version, so degrade to empty rather than crash."""
    fields: dict[str, dict] = {}
    if not isinstance(analysis_result, dict):
        return {"fields": fields, "markdown": "", "usage": {}}

    result = analysis_result.get("result", analysis_result)
    contents = result.get("contents") or []
    markdown_parts = []
    for content in contents:
        md = content.get("markdown")
        if md:
            markdown_parts.append(md)
        page = _first_page(content)
        for name, spec in (content.get("fields") or {}).items():
            if not isinstance(spec, dict):
                continue
            fields[name] = {
                "value": _field_value(spec),
                "confidence": float(spec.get("confidence", 0.5) or 0.0),
                "source_page": page,
            }
    return {"fields": fields, "markdown": "\n\n".join(markdown_parts), "usage": {}}


def _field_value(spec: dict) -> object:
    for key in ("valueString", "valueNumber", "valueDate", "valueBoolean", "value"):
        if key in spec and spec[key] is not None:
            return spec[key]
    return spec.get("content")


def _first_page(content: dict) -> int:
    spans = content.get("pages") or content.get("pageNumbers") or []
    if spans and isinstance(spans, list):
        first = spans[0]
        if isinstance(first, dict):
            return int(first.get("pageNumber", 1))
        if isinstance(first, (int, float)):
            return int(first)
    return 1
