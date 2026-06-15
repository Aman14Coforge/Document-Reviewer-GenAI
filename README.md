# Document Reviewer — GDP Compliance Checker

Automated compliance checking for PDF, DOCX, and TXT documents against GDP (Good Documentation Practice) rules.

> **Cheat sheet:** [CHEATSHEET.md](CHEATSHEET.md) · **Full flow:** [docs/FLOW.md](docs/FLOW.md)

The pipeline validates uploads, extracts text (native or OCR), optionally chunks by section, retrieves relevant rules via vector search, and uses an LLM to evaluate each rule as **passed**, **failed**, **not_applicable**, or **insufficient_evidence**.

---

## Table of Contents

- [Cheat Sheet](#cheat-sheet)
- [Overview](#overview)
- [Architecture](#architecture)
- [Compliance Modes](#compliance-modes)
- [Local vs Office](#local-vs-office)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [One-Time: Embed Rules (RAG)](#one-time-embed-rules-rag)
- [Run Compliance Checks](#run-compliance-checks)
- [Streamlit UI](#streamlit-ui)
- [Scripts Reference](#scripts-reference)
- [Configuration](#configuration)
- [Output Formats](#output-formats)
- [GDP Rules](#gdp-rules)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)

---

## Cheat Sheet

See **[CHEATSHEET.md](CHEATSHEET.md)** for commands, Streamlit, LLM flags, outputs, and troubleshooting in one page.

---

## Overview

This project checks regulatory-style documents (SOPs, deployment reports, protocols) against **13 GDP compliance rules** covering headers, revision history, approvals, formatting, required sections, and page numbers.

**Current iteration:** LLM-based compliance with two modes — whole-document (all rules) and chunk + RAG (vector-matched rules per section).

---

## Architecture

```
rules/rules.json ──► embed_rules.py ──► Chroma (data/chroma/)
                                              │
PDF / DOCX / TXT ──► validate ──► extract ──► chunk ──► match rules (RAG)
                                              │              │
                                              └──────────────┼──► LLM ──► report.json
                                                             │
                              whole_doc mode: full_text + all 13 rules ──► LLM ──► report.json
```

### LLM prompts

| Prompt | File | Used by |
|--------|------|---------|
| **Prompt 1** | `prompts/rules_to_json.txt` | `rules_from_text.py` — plain rules → JSON |
| **Whole doc** | `prompts/compliance_check_whole_doc.txt` | Flow 1 — all 13 rules + full text |
| **Chunk** | `prompts/compliance_check_chunk.txt` | Flow 2 — matched rules + chunk text |
| **Legacy** | `prompts/compliance_check.txt` | `check_compliance.py` CLI (older `--mode` flags) |

The 13 GDP rules are already in `rules/rules.json`. Run Prompt 1 only when new plain-language rules arrive.

---

## Compliance Modes

| Mode | Flow | Rules sent to LLM | Best for |
|------|------|---------------------|----------|
| **Whole doc** (`whole_doc`) | Extract → full text + **all 13 rules** → one LLM call | All rules every time | Simpler runs, short docs, office LLM testing |
| **Chunk + RAG** (`chunk_rag`) | Extract → chunk → Chroma top-k match → per-chunk LLM → merge | ~5 retrieved + 3 always-included per chunk | Scaling when rule sets grow |

**Merge logic (chunk mode):** if any chunk marks a rule **failed** → final status is **failed**. Evidence from multiple chunks is combined.

---

## Local vs Office

Same codebase on both machines; swap the “brain” services via config and flags.

| Component | Here (local dev) | Office (production test) |
|-----------|------------------|---------------------------|
| **LLM** | Dummy (keyword heuristics) or Model Garden | Model Garden via `.env` |
| **Embeddings** | Local `all-MiniLM-L6-v2` + Chroma | Office embedding API *(not wired yet)* |
| **Goal** | Test pipeline, chunking, RAG wiring, UI | Real compliance quality |

**Important:** Re-run `embed_rules.py` on office if the embedding model differs from local MiniLM. Do not copy `data/chroma/` across environments unless the same model is used.

---

## Project Structure

```
document reviewer/
├── config/
│   ├── validation.json            # File size, type, error messages
│   ├── compliance.json            # RAG top-k, prompts, always-include rules
│   └── ocr.json                   # EasyOCR settings
├── data/
│   └── chroma/                    # Persisted rule vectors (after embed_rules.py)
├── docs/                          # Sample documents + FLOW.md
├── rules/
│   ├── rules.json                 # 13 structured GDP rules
│   └── rules_plain.txt
├── prompts/
│   ├── rules_to_json.txt
│   ├── compliance_check_whole_doc.txt
│   ├── compliance_check_chunk.txt
│   └── compliance_check.txt       # Legacy
├── scripts/
│   ├── validate_document.py
│   ├── extract_document.py
│   ├── chunk_document.py
│   ├── embed_rules.py             # One-time: rules → Chroma
│   ├── match_chunk_rules.py       # Standalone chunk → rule matching
│   ├── check_compliance.py
│   ├── rules_from_text.py
│   ├── run_whole_doc_check.py     # Flow 1 end-to-end
│   └── run_chunk_rag_check.py     # Flow 2 end-to-end
├── src/
│   ├── compliance_pipeline.py     # Full pipeline for CLI + Streamlit
│   ├── llm_client.py              # Model Garden + dummy LLM
│   ├── output_utils.py            # Timestamped output paths
│   ├── rag/
│   │   ├── rule_store.py          # Chroma embed + retrieve
│   │   └── chunk_matcher.py       # Match rules per chunk
│   └── ocr/                       # EasyOCR + cloud stub
├── output/
│   ├── validation/
│   ├── extracted/
│   ├── chunks/
│   ├── matches/                   # chunk → rules JSON (RAG)
│   ├── reports/
│   ├── rules/
│   └── uploads/                   # Streamlit uploads
├── main.py                        # Validate → extract → chunk only
├── streamlit_app.py               # Web UI
├── requirements.txt
├── .env.example
└── README.md
```

---

## Prerequisites

- **Python 3.10+**
- **Model Garden** credentials (office / real LLM runs only)
- Supported inputs: **`.pdf`**, **`.docx`**, **`.txt`**
- For OCR: EasyOCR (first run downloads models)
- For RAG: `chromadb`, `sentence-transformers` (included in `requirements.txt`)

---

## Setup

```powershell
cd "c:\Users\user\Desktop\document reviewer"
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` for Model Garden (office / real LLM):

```env
MODEL_GARDEN_BASE_URL=https://your-model-garden-gateway-url
MODEL_GARDEN_API_KEY=your_api_key_here
LLM_MODEL=your_model_name_here
```

The LLM client sends the API key as an `X-API-KEY` header.

---

## One-Time: Embed Rules (RAG)

Required before **chunk + RAG** mode. Embeds all rules from `rules/rules.json` into Chroma using `all-MiniLM-L6-v2`.

```powershell
python scripts/embed_rules.py --rebuild
```

When rules change, edit `rules/rules.json` and rerun:

```powershell
python scripts/embed_rules.py          # upsert new/changed rules
python scripts/embed_rules.py --rebuild   # wipe and re-index
```

Output: `data/chroma/` + `data/chroma/last_embed_summary.json`

---

## Run Compliance Checks

### Flow 1 — Whole document (all 13 rules → LLM)

```powershell
python scripts/run_whole_doc_check.py "docs/Deployment Report_v0 - filled.pdf"
python scripts/run_whole_doc_check.py "docs/your_doc.pdf" --extraction ocr --llm model_garden
```

### Flow 2 — Chunk + RAG (vector match → per-chunk LLM)

```powershell
python scripts/embed_rules.py --rebuild
python scripts/run_chunk_rag_check.py "docs/Deployment Report_v0 - filled.pdf"
python scripts/run_chunk_rag_check.py "docs/your_doc.pdf" --llm dummy
```

### Extract + chunk only (no compliance)

```powershell
python main.py
```

Edit `DOCUMENT_PATH` and `EXTRACTION_MODE` (`native` or `ocr`) at the top of `main.py`.

### Manual step-by-step

```powershell
python scripts/extract_document.py "docs/Deployment Report_v0 - filled.pdf"
python scripts/chunk_document.py "output/extracted/Deployment Report_v0 - filled_<timestamp>_extracted.json"
python scripts/match_chunk_rules.py "output/chunks/Deployment Report_v0 - filled_<timestamp>_chunks.json"
python scripts/check_compliance.py "output/chunks/..." --mode per-chunk --file-name "Deployment Report_v0 - filled.pdf"
```

All outputs use **timestamps** in filenames — nothing is overwritten between runs.

---

## Streamlit UI

```powershell
python -m streamlit run streamlit_app.py
```

**Upload & Check tab:**
- Upload document (rules upload optional, JSON only wired)
- Extraction: native or OCR
- Compliance mode: whole doc or chunk + RAG
- LLM engine: dummy (local) or Model Garden

**Results tab:** overall status, rule table, chunk→rule matches (RAG), download JSON.

---

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `validate_document.py` | File type, size, readability, empty content |
| `extract_document.py` | PDF/DOCX/TXT → JSON (`--mode native\|ocr`) |
| `chunk_document.py` | Section-wise chunks from extracted JSON |
| `embed_rules.py` | Rules → Chroma vector store |
| `match_chunk_rules.py` | Chunks → matched rules JSON |
| `check_compliance.py` | LLM compliance (legacy CLI modes) |
| `run_whole_doc_check.py` | Full Flow 1 pipeline |
| `run_chunk_rag_check.py` | Full Flow 2 pipeline |
| `rules_from_text.py` | Plain rules → JSON via LLM (Prompt 1) |
| `main.py` | Validate → extract → chunk |

---

## Configuration

| File | Purpose |
|------|---------|
| `config/validation.json` | Max size (50 MB), allowed types, error messages |
| `config/compliance.json` | Embedding model, RAG top-k, always-include rule IDs, prompt paths |
| `config/ocr.json` | EasyOCR languages, DPI, page-chunk fallback |
| `.env` | Model Garden URL, API key, model name |

**RAG defaults** (`config/compliance.json`):

| Setting | Default | Description |
|---------|---------|-------------|
| `top_k_rules_per_chunk` | `5` | Vector search results per chunk |
| `always_include_rule_ids` | GDP-08, 09, 10 | Always added to each chunk’s rule set |
| `skip_chunk_ids` | `full_document` | Skip redundant full-doc chunk in RAG |

---

## Output Formats

### Compliance report (`output/reports/`)

```json
{
  "file_name": "Deployment Report_v0 - filled.pdf",
  "mode": "chunk_rag",
  "rule_retrieval": "rag",
  "chunk_rule_matches": { "matches": [ ... ] },
  "summary": {
    "overall_status": "compliant | non_compliant | needs_review",
    "passed": 8,
    "failed": 3,
    "total_rules": 13
  },
  "results": [ { "rule_id": "GDP-01", "status": "passed", ... } ]
}
```

### Chunk rule matches (`output/matches/`)

Per chunk: `retrieved_rule_ids` (from Chroma), `matched_rule_ids` (+ always-include), full `matched_rules` objects sent to LLM.

**Overall status:**

| Status | Meaning |
|--------|---------|
| `compliant` | All rules passed or N/A |
| `non_compliant` | At least one rule failed |
| `needs_review` | No failures, but insufficient evidence on some rules |

---

## GDP Rules

All 13 rules in `rules/rules.json`:

| Rule ID | Title | Type |
|---------|-------|------|
| GDP-01 | Document Title on First Page Matches File Name | semantic |
| GDP-02 | Author Name and Role Are Mentioned | semantic |
| GDP-03 | Dates in Revision History Are Valid | deterministic |
| GDP-04 | Version Number Present in Title and Revision History | deterministic |
| GDP-05 | Revision Section Is Present | deterministic |
| GDP-06 | Signature Blocks Are Present | deterministic |
| GDP-07 | Dates Present Near Signatures | deterministic |
| GDP-08 | Basic Font and Spacing Consistency | semantic |
| GDP-09 | Language Errors Identified | semantic |
| GDP-10 | Required Sections Present | semantic |
| GDP-11 | Page Numbers Present and Sequential | deterministic |
| GDP-12 | Readability Within Acceptable Range | semantic |
| GDP-13 | Footer Contains Doc ID, Page Number, Confidentiality | deterministic |

---

## Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| Validation + extract + chunk | Done | Native + OCR paths |
| Whole-doc LLM compliance | Done | All 13 rules, one call |
| Chunk + RAG compliance | Done | Chroma + per-chunk LLM + merge |
| Streamlit UI | Done | Upload, modes, results |
| Office embedding provider | Planned | Swap local MiniLM for office API |
| Rules upload flow | Planned | Plain rules → Prompt 1 → use in check |
| Rule routing by scope | Planned | Deterministic / chunk / whole-doc per rule |
| Doc-type filtering | Planned | 100 doc types × ~20 rules each |
| Regex checks (Iteration 1) | Planned | Deterministic rules without LLM |

---

## Troubleshooting

### `Rule vector store is empty`

Run embedding first:

```powershell
python scripts/embed_rules.py --rebuild
```

### Dummy LLM returns empty chunk results

The chunk prompt uses `MATCHED RULES FOR THIS CHUNK:`; the dummy parser expects `RULES TO EVALUATE:`. Use **whole doc mode** for local dummy testing, or **Model Garden** for chunk mode. Fix planned in `llm_client.py`.

### `MODEL_GARDEN_API_KEY is not set`

Only needed for `--llm model_garden` or Streamlit Model Garden option. Dummy mode works without `.env`.

### OCR slow on first run

EasyOCR downloads models on first use. Set `EXTRACTION_MODE = "native"` for digital PDFs.

### Streamlit not found

Use:

```powershell
python -m streamlit run streamlit_app.py
```

### GDP-01 needs file name

Pass `--file-name "Your Document.pdf"` or use `run_whole_doc_check.py` / `run_chunk_rag_check.py` (handles this automatically).

---

## Quick Reference

```powershell
pip install -r requirements.txt
python scripts/embed_rules.py --rebuild

# Flow 1 — whole doc, dummy LLM
python scripts/run_whole_doc_check.py "docs/Deployment Report_v0 - filled.pdf"

# Flow 2 — chunk + RAG, dummy LLM
python scripts/run_chunk_rag_check.py "docs/Deployment Report_v0 - filled.pdf"

# Office — real LLM
python scripts/run_whole_doc_check.py "docs/your_doc.pdf" --llm model_garden

# UI
python -m streamlit run streamlit_app.py

# Extract pipeline only
python main.py
```
