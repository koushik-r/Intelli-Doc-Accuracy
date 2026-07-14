# Intelli-Doc Accuracy — Backend

High-accuracy document extraction with **two independent extraction paths**
(LLM via Azure AI Foundry, and Azure Content Understanding). The paths never
depend on each other — call whichever one you select.

## Structure

```
Backend/
  models/
    template.py           # ExtractionTemplate / FieldSpec (the field set to extract)
    extraction_result.py  # ExtractedField / ExtractionResult
    usage.py              # Usage (token + batch cost tracking)
  extraction/
    ACU/
      Acu.py               # ACU path: client + overlap batching + merge + schema
    LLM/
      LLM.py               # LLM path: layout + Foundry client + prompts + extractor
    Templates/
      Template.py          # infer a template from a sample document
  main.py                  # entry point
```

## Setup

```bash
uv sync
cp .env.example .env       # fill in Azure AI Foundry and/or ACU credentials
```

## Usage

The three paths are plain importable functions:

```python
from extraction.LLM.LLM import extract_document_structured, extract_fields_llm, get_model_client
from extraction.ACU.Acu import ACUClient, run_acu_with_batching, merge_acu_batches
from extraction.Templates.Template import generate_template
from models.template import FNOL_TEMPLATE

pdf = open("doc.pdf", "rb").read()

# LLM path (needs AZURE_AI_FOUNDRY_* in .env)
structured = extract_document_structured(pdf)
llm_result = extract_fields_llm(structured, FNOL_TEMPLATE, get_model_client())

# Template generation (offline heuristic if no client passed)
template, usage = generate_template(pdf, template_id="auto", name="Auto")
```

The LLM path uses **Azure AI Foundry** (OpenAI-compatible chat completions),
configured via `AZURE_AI_FOUNDRY_*`. ACU uses `AZURE_CONTENT_UNDERSTANDING_*`.

## Architecture decisions

### Structured layout extraction (no docling)

Flat-text PDF extraction destroys the two things an LLM needs to be accurate on
mixed-layout documents: **table structure** and **field boundaries**. We rebuild
both from geometry using `pdfplumber` (PyMuPDF is the fallback for pages with no
text layer).

Three decisions that make it correct on messy real documents:

1. **Incremental confirmed bboxes.** Only a table that passes
   `_is_plausible_table()` (≥2 rows, ≥2 non-empty columns) contributes its bbox
   to the "confirmed" list. A *rejected* false-positive table must **not** mask
   its region during the narrative pass — otherwise its words are dropped from
   both the table output and the narrative output.

2. **Full 2D bbox overlap, not a y-range test.** A word is assigned to a table
   only if >50% of *the word's own area* lies inside a confirmed table bbox
   (`_word_in_bbox`). A y-range-only test wrongly swallows a right-margin sidebar
   that merely shares a table's vertical band but never overlaps in x.

3. **Column breaks preserved in narrative.** Words are grouped into lines by
   rounded y-position; wherever the x-gap between consecutive words exceeds 15pt
   we insert a `|`. This keeps distinct fields separate
   (`Date of Occurrence 11/18/2025 | Time: 2:45 PM`) so the LLM never merges two
   fields into one.

Pages are tagged with `<!-- PAGE N -->` / `<!-- END PAGE N -->` anchors for
source attribution, and classified `digital` vs `scanned` by text density
(scanned pages are a TODO OCR route).

### Template generation

If a user has **no** predefined template, they upload a sample document and we
infer one. `Templates/Template.py` runs the structured extraction over the
document, then proposes an `ExtractionTemplate` (a field set):

* **LLM engine** (`generate_template_llm`) — asks the model to propose reusable
  field definitions from the structured content. Best quality; needs Foundry.
* **Heuristic engine** (`generate_template_heuristic`) — parses `Label: value`
  pairs and 2-column table rows straight from the structured pages, infers each
  field's type (date/number/bool/string) and sets `prefers_table` for
  grid-sourced fields. **No LLM required**, so it works offline.

`generate_template()` uses the LLM engine when a client is passed and falls back
to the heuristic otherwise.

### LLM extraction

Chunks the structured pages (whole pages, anchors intact) under a token budget,
then runs **two prompts per chunk**: a strict low-temperature pass over the
table markdown (exact label/value matching) and a permissive pass over the
narrative prose (semantic entities). Each value is tagged with its source
(`table` vs `narrative`). `merge_chunk_results` prefers higher confidence and,
on a near-tie, the table-sourced value. The provider is abstracted behind a
`ModelClient` protocol with an Azure AI Foundry implementation, so extraction
logic never touches credentials.

### ACU client + overlap batching

`ACUClient.analyze_document` polls `Operation-Location` and, on failure, raises a
typed `ACUAnalysisError` with the real status/body — never swallows to `None`.
For documents over the ~150-page / 100 MB limit, `run_acu_with_batching` splits
into **overlapping** batches (each with a `core_end` marking its trusted region),
injects a page-1 "document identity" context into every call, and keeps going if
a single batch fails. `merge_acu_batches` de-dups the overlap zones and, on
disagreement, trusts the batch holding a page in its **core** range over the one
holding it in its overlap tail — content near a hard boundary is more likely a
fragment. The client is injected, so a mock and the real client are
interchangeable.
