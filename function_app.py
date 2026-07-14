"""Azure Functions (Python v2) HTTP API for the IDP extraction backend.

This is the *second* backend the Angular frontend talks to (the first handles
plain uploads). Everything here is namespaced under `/api/idp/...` and covers:

  * documents  — upload (to Blob) + run extraction, list, get, preview, delete
  * templates  — infer from a sample doc + store, list, get
  * extractions — extraction result + per-field confidence for a document
  * usage      — token/cost usage per document and aggregated

CORS is handled by host.json + the Functions runtime for local dev; each
response also carries permissive headers so `ng serve` can call it directly.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

import azure.functions as func
from bson import ObjectId
from bson.errors import InvalidId

from files import get_blob_store
from datasource import collections, ensure_indexes, get_db
from extraction.LLM.LLM import (
    ModelClientError,
    extract_document_structured,
    extract_fields_llm,
    get_model_client,
)
from extraction.Templates.Template import generate_template
from models.extraction_result import ExtractionResult
from models.files import (
    DocumentRecord,
    DocumentStatus,
    ExtractionResultRecord,
    TemplateRecord,
    UsageRecord,
)
from models.template import ExtractionTemplate, get_template

logger = logging.getLogger(__name__)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

_CORS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,DELETE,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,Authorization",
}


def _json(payload: Any, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload, default=str),
        status_code=status,
        mimetype="application/json",
        headers=_CORS,
    )


def _error(message: str, status: int = 400) -> func.HttpResponse:
    return _json({"error": message}, status)


# ==========================================================================
# Orchestration — blob + extraction + persistence glue
# ==========================================================================
def _overall_confidence(result: ExtractionResult) -> float:
    """Mean confidence over the fields that actually got a value."""
    vals = [f.confidence for f in result.fields.values() if f.value not in (None, "")]
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _resolve_template(template_id: str | None) -> ExtractionTemplate:
    """Load a template: prefer a stored one, fall back to the builtins."""
    if template_id:
        stored = get_db()[collections.templates].find_one({"template_id": template_id})
        rec = TemplateRecord.from_mongo(stored)
        if rec is not None:
            return rec.definition
        return get_template(template_id)
    return get_template("generic")


def run_extraction_for_document(document_id: str, template_id: str | None = None) -> DocumentRecord:
    """Run the LLM extraction path for an already-uploaded document, persisting
    the result + usage and updating the document record."""
    ensure_indexes()
    db = get_db()
    docs = db[collections.documents]

    doc = DocumentRecord.from_mongo(docs.find_one({"_id": ObjectId(document_id)}))
    if doc is None:
        raise KeyError(f"Document '{document_id}' not found")

    docs.update_one(
        {"_id": ObjectId(document_id)}, {"$set": {"status": DocumentStatus.PROCESSING.value}}
    )

    pdf_bytes = get_blob_store().download(doc.blob_name)

    try:
        template = _resolve_template(template_id or doc.template_id)
        client = get_model_client()

        started = time.perf_counter()
        structured = extract_document_structured(pdf_bytes)
        result = extract_fields_llm(structured, template, client)
        result.usage.elapsed_seconds = round(time.perf_counter() - started, 3)
    except Exception as exc:
        logger.exception("Extraction failed for %s", document_id)
        docs.update_one(
            {"_id": ObjectId(document_id)},
            {"$set": {"status": DocumentStatus.FAILED.value, "error": str(exc)}},
        )
        raise

    conf = _overall_confidence(result)

    result_rec = ExtractionResultRecord(
        document_id=document_id,
        template_id=template.template_id,
        method=result.method.value,
        overall_confidence=conf,
        result=result,
    )
    result_id = str(db[collections.extraction_results].insert_one(result_rec.to_mongo()).inserted_id)

    usage_rec = UsageRecord(
        document_id=document_id,
        extraction_result_id=result_id,
        template_id=template.template_id,
        method=result.method.value,
        usage=result.usage,
    )
    db[collections.usage].insert_one(usage_rec.to_mongo())

    docs.update_one(
        {"_id": ObjectId(document_id)},
        {
            "$set": {
                "status": DocumentStatus.EXTRACTED.value,
                "template_id": template.template_id,
                "extraction_result_id": result_id,
                "overall_confidence": conf,
                "error": None,
            }
        },
    )
    return DocumentRecord.from_mongo(docs.find_one({"_id": ObjectId(document_id)}))


def infer_and_store_template(
    pdf_bytes: bytes,
    template_id: str,
    name: str | None = None,
    sample_document_id: str | None = None,
    use_llm: bool = True,
) -> TemplateRecord:
    """Infer a template from a sample document and store it in Mongo."""
    ensure_indexes()
    client = None
    if use_llm:
        try:
            client = get_model_client()
        except ModelClientError:
            client = None  # fall back to the offline heuristic

    template = generate_template(pdf_bytes, template_id, name, client=client)[0]

    rec = TemplateRecord(
        template_id=template.template_id,
        name=template.name,
        description=template.description,
        source="inferred" if client else "manual",
        definition=template,
        sample_document_id=sample_document_id,
    )
    db = get_db()
    db[collections.templates].update_one(
        {"template_id": template.template_id}, {"$set": rec.to_mongo()}, upsert=True
    )
    return TemplateRecord.from_mongo(
        db[collections.templates].find_one({"template_id": template.template_id})
    )


def _oid(value: str) -> ObjectId:
    try:
        return ObjectId(value)
    except (InvalidId, TypeError):
        raise ValueError(f"Invalid id '{value}'")


# ==========================================================================
# Documents — upload + extract, list, get, preview, delete
# ==========================================================================
@app.route(route="idp/documents/upload", methods=["POST", "OPTIONS"])
def upload_document(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)

    ensure_indexes()
    files = req.files.get("file")
    if files is None:
        return _error("No 'file' part in multipart form data.")

    filename = getattr(files, "filename", "document")
    content_type = getattr(files, "content_type", None) or "application/octet-stream"
    data = files.stream.read()
    if not data:
        return _error("Uploaded file is empty.")

    template_id = req.params.get("template_id") or (req.form.get("template_id") if req.form else None)
    run_now = (req.params.get("extract", "true").lower() != "false")

    blob = get_blob_store()
    blob_name, blob_url = blob.upload(data, filename, content_type)

    rec = DocumentRecord(
        original_filename=filename,
        content_type=content_type,
        size_bytes=len(data),
        blob_container=blob.container,
        blob_name=blob_name,
        blob_url=blob_url,
        status=DocumentStatus.UPLOADED,
        template_id=template_id,
    )
    db = get_db()
    doc_id = str(db[collections.documents].insert_one(rec.to_mongo()).inserted_id)

    if run_now:
        try:
            updated = run_extraction_for_document(doc_id, template_id)
            return _json(updated.model_dump(by_alias=True), 201)
        except Exception as exc:  # extraction failure — doc is still stored
            logger.exception("Extraction failed during upload")
            stored = DocumentRecord.from_mongo(
                db[collections.documents].find_one({"_id": _oid(doc_id)})
            )
            body = stored.model_dump(by_alias=True) if stored else {"_id": doc_id}
            body["extraction_error"] = str(exc)
            return _json(body, 201)

    stored = DocumentRecord.from_mongo(db[collections.documents].find_one({"_id": _oid(doc_id)}))
    return _json(stored.model_dump(by_alias=True), 201)


@app.route(route="idp/documents", methods=["GET", "OPTIONS"])
def list_documents(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    db = get_db()
    cursor = db[collections.documents].find().sort("created_at", -1).limit(500)
    items = [DocumentRecord.from_mongo(d).model_dump(by_alias=True) for d in cursor]
    return _json(items)


@app.route(route="idp/documents/{id}", methods=["GET", "DELETE", "OPTIONS"])
def document_by_id(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    doc_id = req.route_params.get("id")
    try:
        oid = _oid(doc_id)
    except ValueError as exc:
        return _error(str(exc))

    db = get_db()
    raw = db[collections.documents].find_one({"_id": oid})
    rec = DocumentRecord.from_mongo(raw)
    if rec is None:
        return _error("Document not found.", 404)

    if req.method == "DELETE":
        try:
            get_blob_store().delete(rec.blob_name)
        except Exception:
            pass
        db[collections.documents].delete_one({"_id": oid})
        db[collections.extraction_results].delete_many({"document_id": doc_id})
        db[collections.usage].delete_many({"document_id": doc_id})
        return _json({"deleted": doc_id})

    return _json(rec.model_dump(by_alias=True))


@app.route(route="idp/documents/{id}/blob", methods=["GET", "OPTIONS"])
def document_blob(req: func.HttpRequest) -> func.HttpResponse:
    """Stream the raw file bytes back for preview/download."""
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    doc_id = req.route_params.get("id")
    try:
        oid = _oid(doc_id)
    except ValueError as exc:
        return _error(str(exc))

    db = get_db()
    rec = DocumentRecord.from_mongo(db[collections.documents].find_one({"_id": oid}))
    if rec is None:
        return _error("Document not found.", 404)

    data = get_blob_store().download(rec.blob_name)
    headers = dict(_CORS)
    headers["Content-Disposition"] = f'inline; filename="{rec.original_filename}"'
    return func.HttpResponse(body=data, status_code=200, mimetype=rec.content_type, headers=headers)


@app.route(route="idp/documents/{id}/extract", methods=["POST", "OPTIONS"])
def reextract_document(req: func.HttpRequest) -> func.HttpResponse:
    """Re-run extraction for an existing document, optionally with a template."""
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    doc_id = req.route_params.get("id")
    template_id = req.params.get("template_id")
    if not template_id and req.get_body():
        try:
            template_id = (req.get_json() or {}).get("template_id")
        except ValueError:
            template_id = None
    try:
        updated = run_extraction_for_document(doc_id, template_id)
    except KeyError:
        return _error("Document not found.", 404)
    except Exception as exc:
        return _error(f"Extraction failed: {exc}", 500)
    return _json(updated.model_dump(by_alias=True))


# ==========================================================================
# Templates — infer from a sample doc + store, list, get
# ==========================================================================
@app.route(route="idp/templates/upload", methods=["POST", "OPTIONS"])
def upload_template(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    files = req.files.get("file")
    if files is None:
        return _error("No 'file' part in multipart form data.")

    data = files.stream.read()
    if not data:
        return _error("Uploaded file is empty.")

    filename = getattr(files, "filename", "sample")
    form = req.form or {}
    template_id = req.params.get("template_id") or form.get("template_id")
    name = req.params.get("name") or form.get("name") or filename
    use_llm = (req.params.get("use_llm", form.get("use_llm", "true")).lower() != "false")

    if not template_id:
        import re

        template_id = re.sub(r"[^a-z0-9]+", "_", filename.lower()).strip("_") or "template"

    try:
        rec = infer_and_store_template(data, template_id, name, use_llm=use_llm)
    except Exception as exc:
        logger.exception("Template inference failed")
        return _error(f"Template inference failed: {exc}", 500)
    return _json(rec.model_dump(by_alias=True), 201)


@app.route(route="idp/templates", methods=["GET", "OPTIONS"])
def list_templates(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    db = get_db()
    cursor = db[collections.templates].find().sort("created_at", -1)
    items = [TemplateRecord.from_mongo(t).model_dump(by_alias=True) for t in cursor]
    return _json(items)


@app.route(route="idp/templates/{template_id}", methods=["GET", "DELETE", "OPTIONS"])
def template_by_id(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    template_id = req.route_params.get("template_id")
    db = get_db()
    raw = db[collections.templates].find_one({"template_id": template_id})
    rec = TemplateRecord.from_mongo(raw)
    if rec is None:
        return _error("Template not found.", 404)
    if req.method == "DELETE":
        db[collections.templates].delete_one({"template_id": template_id})
        return _json({"deleted": template_id})
    return _json(rec.model_dump(by_alias=True))


# ==========================================================================
# Extractions — result + per-field confidence
# ==========================================================================
@app.route(route="idp/extractions/{document_id}", methods=["GET", "OPTIONS"])
def extraction_for_document(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    document_id = req.route_params.get("document_id")
    db = get_db()
    raw = (
        db[collections.extraction_results]
        .find({"document_id": document_id})
        .sort("created_at", -1)
        .limit(1)
    )
    docs = list(raw)
    if not docs:
        return _error("No extraction result for this document.", 404)
    rec = ExtractionResultRecord.from_mongo(docs[0])
    return _json(rec.model_dump(by_alias=True))


# ==========================================================================
# Usage — per document and aggregated
# ==========================================================================
@app.route(route="idp/usage", methods=["GET", "OPTIONS"])
def usage_summary(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "OPTIONS":
        return func.HttpResponse(status_code=204, headers=_CORS)
    db = get_db()
    document_id = req.params.get("document_id")
    query = {"document_id": document_id} if document_id else {}
    records = [UsageRecord.from_mongo(u) for u in db[collections.usage].find(query)]

    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
        "api_calls": 0,
        "runs": len(records),
    }
    for r in records:
        totals["input_tokens"] += r.usage.input_tokens
        totals["output_tokens"] += r.usage.output_tokens
        totals["total_tokens"] += r.usage.total_tokens
        totals["api_calls"] += r.usage.api_calls

    return _json(
        {
            "totals": totals,
            "records": [r.model_dump(by_alias=True) for r in records],
        }
    )
