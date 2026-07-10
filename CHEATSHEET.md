# Document Reviewer — Cheat Sheet

Quick reference for daily use. Full docs: [README.md](README.md) · [docs/FLOW.md](docs/FLOW.md)

---

## First-time setup

```powershell
cd "c:\Users\user\Desktop\document reviewer"
pip install -r requirements.txt
copy .env.example .env
python scripts/embed_rules.py --rebuild
```

Edit `.env` for **Model Garden** (office):

```env
MODEL_GARDEN_BASE_URL=https://your-gateway-url
MODEL_GARDEN_API_KEY=your_key
LLM_MODEL=your_model_name

# Embeddings (chunk + RAG)
EMBEDDING_PROVIDER=model_garden
EMBEDDING_API_URL=https://your-gateway/.../embeddings
EMBEDDING_MODEL=your-embed-model
EMBEDDING_DIM=746

# Optional: debug logging
LOG_LEVEL=INFO
```

Local dev — keep `EMBEDDING_PROVIDER=local` and put MiniLM in `model/` folder.

---

## Rules overview (20 total)

| Type | Count | Rules | Engine |
|------|-------|-------|--------|
| **semantic** | 10 | GDP-01–07, 12, 15, 16 | LLM (chunk RAG or whole doc) |
| **deterministic** | 7 | GDP-08–11, 13, 14, 17 | Python (`src/deterministic_checker.py`) |
| **existential** | 3 | GDP-18–20 | Python + external JSON (`src/existential_checker.py`) |

**Hybrid mode** (default): semantic rules go to the LLM; deterministic and existential rules run in Python — no LLM cost for those.

Files: `rules/rules.json` · plain source: `rules/rules_plain.txt`

---

## Streamlit UI

```powershell
python -m streamlit run streamlit_app.py
```

| Setting | Options |
|---------|---------|
| Extraction | `native` (digital PDF) · `ocr` (scanned PDF) |
| Compliance | `whole_doc` (all rules, hybrid) · `chunk_rag` (vector match + hybrid) |
| LLM | `dummy` (no `.env`) · `model_garden` (needs `.env`) |

**Chunk + RAG requires embed first:** `python scripts/embed_rules.py --rebuild`

Both modes use hybrid checking — deterministic and existential rules always run in Python.

---

## Compliance flows (CLI)

### Flow 1 — Whole document (hybrid)

Full text → semantic rules via LLM + deterministic/existential via Python.

```powershell
python scripts/run_whole_doc_check.py "docs/your_doc.pdf"
python scripts/run_whole_doc_check.py "docs/your_doc.pdf" --extraction ocr
python scripts/run_whole_doc_check.py "docs/your_doc.pdf" --llm model_garden
```

Or use the dedicated hybrid script:

```powershell
python scripts/run_hybrid_check.py "docs/your_doc.pdf" --llm model_garden
python scripts/run_hybrid_check.py "docs/your_doc.pdf" --llm dummy
python scripts/run_hybrid_check.py "docs/your_doc.pdf" --llm-only   # disable Python checks
```

### Flow 2 — Chunk + RAG (hybrid)

Chunk → Chroma match (semantic only) → per-chunk LLM → aggregate → merge with deterministic/existential.

```powershell
python scripts/embed_rules.py --rebuild
python scripts/run_chunk_rag_check.py "docs/your_doc.pdf"
python scripts/run_chunk_rag_check.py "docs/your_doc.pdf" --llm model_garden
```

**Aggregation:** a semantic rule **passes** if **≥2 chunks** return `passed` (`config/compliance.json` → `aggregation.pass_if_min_chunk_passes`).

**RAG:** top **2** semantic rules per chunk, keyword rerank enabled, deterministic/existential rules never sent to LLM.

---

## Extract + chunk only (no compliance)

Edit top of `main.py`:

```python
DOCUMENT_PATH = PROJECT_ROOT / "docs" / "your_doc.pdf"
EXTRACTION_MODE = "native"   # or "ocr"
```

```powershell
python main.py
```

---

## Manual step-by-step

```powershell
# 1. Validate (optional standalone)
python scripts/validate_document.py

# 2. Extract
python scripts/extract_document.py "docs/your_doc.pdf"
python scripts/extract_document.py "docs/scanned.pdf" --mode ocr

# 3. Chunk
python scripts/chunk_document.py "output/extracted/your_doc_<timestamp>_extracted.json"

# 4. Match rules (RAG — semantic rules only)
python scripts/match_chunk_rules.py "output/chunks/your_doc_<timestamp>_chunks.json"

# 5. Compliance (hybrid)
python scripts/check_compliance.py "output/chunks/..." --file-name "your_doc.pdf"
```

---

## Rules & RAG

### Text → JSON → embed (full workflow)

**Step 1 — Write or edit plain rules** in `rules/rules_plain.txt` (one rule per block, GDP-XX IDs).

**Step 2 — Convert text to JSON** (needs Model Garden in `.env` — `MODEL_GARDEN_BASE_URL`, `MODEL_GARDEN_API_KEY`, `LLM_MODEL`). Response is validated with **Pydantic** (`src/llm_schemas.py`).

```powershell
# Write to default rules file (used by pipeline + embed)
python scripts/rules_from_text.py rules/rules_plain.txt -o rules/rules.json

# Or save a timestamped copy (default if you omit -o)
python scripts/rules_from_text.py rules/rules_plain.txt
# → output/rules/generated_rules_<timestamp>.json
```

Other input options:

```powershell
python scripts/rules_from_text.py --text "GDP-01: Title must match file name." -o rules/rules.json
type rules\rules_plain.txt | python scripts/rules_from_text.py - -o rules/rules.json
```

**Step 3 — Embed into Chroma** (uses `EMBEDDING_PROVIDER` from `.env` — `local` or `model_garden`):

```powershell
# After replacing rules/rules.json
python scripts/embed_rules.py --rebuild

# Or embed a generated file without overwriting rules/rules.json
python scripts/embed_rules.py --rules output/rules/generated_rules_20260630_112055.json --rebuild
```

Use `--rebuild` when rule IDs or embedding model changed. Use without `--rebuild` for small edits (upsert only).

**Step 4 — Run chunk + RAG** (only semantic rules are retrieved from Chroma; deterministic/existential still run in Python):

```powershell
python scripts/run_chunk_rag_check.py "docs/your_doc.pdf" --llm model_garden
```

Copy-paste one-liner (text → default rules → embed):

```powershell
python scripts/rules_from_text.py rules/rules_plain.txt -o rules/rules.json
python scripts/embed_rules.py --rebuild
```

### Quick reference

| Task | Command |
|------|---------|
| Plain text → JSON | `python scripts/rules_from_text.py rules/rules_plain.txt -o rules/rules.json` |
| Embed rules into Chroma | `python scripts/embed_rules.py --rebuild` |
| Embed a specific JSON file | `python scripts/embed_rules.py --rules path/to/rules.json --rebuild` |
| Upsert after small JSON edit | `python scripts/embed_rules.py` |

Chroma data: `data/chroma/` · Default rules: `rules/rules.json` · Plain source: `rules/rules_plain.txt`

---

## External dependencies (existential rules)

GDP-18, GDP-19, GDP-20 check against JSON registries:

| File | Used for |
|------|----------|
| `data/external/audit_log.json` | GDP-18 — audit trail entries |
| `data/external/reference_registry.json` | GDP-19 — referenced document IDs |
| `data/external/traceability_matrix.json` | GDP-20 — URS traceability |

Paths configured in `config/compliance.json` → `external_dependencies`.

---

## LLM quick pick

| Where | Local test | Real LLM |
|-------|------------|----------|
| Streamlit | LLM engine → **Dummy** | LLM engine → **Model Garden** |
| CLI whole doc | `--llm dummy` | `--llm model_garden` |
| CLI chunk RAG | `--llm dummy` (default) | `--llm model_garden` |
| CLI hybrid | `--llm dummy` | `--llm model_garden` (default) |

Dummy = keyword heuristics, no API. **Not for sign-off** — use Model Garden for real checks.

---

## Output folders

| Folder | What |
|--------|------|
| `output/validation/` | Pass/fail checks |
| `output/extracted/` | Page text + `full_text` |
| `output/chunks/` | Section chunks |
| `output/matches/` | Chunk → matched rules (RAG) |
| `output/reports/` | Final compliance report |
| `output/uploads/` | Streamlit uploads |
| `output/logs/` | `compliance.log` (set `LOG_LEVEL=DEBUG` for detail) |
| `data/chroma/` | Embedded rule vectors |
| `data/external/` | Audit log, reference registry, traceability matrix |

Filenames include a **timestamp** — runs never overwrite each other.

---

## Config files

| File | Change what |
|------|-------------|
| `config/validation.json` | Max size, allowed types, error messages |
| `config/compliance.json` | RAG top-k, aggregation, hybrid, external paths |
| `config/ocr.json` | EasyOCR languages, DPI |
| `.env` | Model Garden URL, key, model, embedding provider |

Key `config/compliance.json` settings:

| Setting | Default | Meaning |
|---------|---------|---------|
| `chunk_rag.top_k_rules_per_chunk` | `2` | Semantic rules retrieved per chunk |
| `chunk_rag.keyword_rerank` | `true` | Boost rules matching chunk keywords |
| `chunk_rag.rag_rule_types` | `["semantic"]` | Only semantic rules go to RAG/LLM |
| `aggregation.pass_if_min_chunk_passes` | `2` | Pass semantic rule if ≥2 chunks pass |
| `whole_doc.hybrid` / `chunk_rag.hybrid` | `true` | Run deterministic + existential in Python |

---

## Local vs office

| Component | Here (dev) | Office |
|-----------|------------|--------|
| LLM | Dummy or Model Garden | Model Garden |
| Embeddings | `local` → `model/` folder | `model_garden` → `EMBEDDING_API_URL` |

Re-run `embed_rules.py` on office if embedding model differs. Don't copy `data/chroma/` across different models.

---

## Report status meanings

| Status | Meaning |
|--------|---------|
| `compliant` | All rules passed or N/A |
| `non_compliant` | At least one rule failed |
| `needs_review` | No failures, but some rules lack evidence |

Per-rule: `passed` · `failed` · `not_applicable` · `insufficient_evidence`

Each result also has `check_method`: `llm` · `deterministic` · `existential`

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `streamlit` not found | `python -m streamlit run streamlit_app.py` |
| Rule vector store empty | `python scripts/embed_rules.py --rebuild` |
| Chunk RAG empty LLM results (dummy) | Use **whole_doc** locally or **model_garden** for chunk mode |
| Model Garden errors | Check `.env` — URL, key, model name |
| OCR slow first run | EasyOCR downloads models; use `native` for digital PDFs |
| Scanned PDF empty (native) | Use `--extraction ocr` or `EXTRACTION_MODE = "ocr"` |
| Existential rule failed | Update `data/external/*.json` with missing doc/URS IDs |
| Semantic passes look inflated (dummy) | Dummy auto-passes many rules — switch to `model_garden` |
| Debug pipeline steps | `$env:LOG_LEVEL="DEBUG"` then re-run; check `output/logs/compliance.log` |

---

## Sample doc

```powershell
python scripts/run_hybrid_check.py "docs/Deployment Report_v0 - filled.pdf" --llm dummy
python scripts/run_chunk_rag_check.py "docs/Deployment Report_v0 - filled.pdf" --llm dummy
python scripts/run_whole_doc_check.py "docs/Deployment Report_v0 - filled.pdf"
```

Test fixture: `test_samples/gdp_test_document.txt`

---

## One-liner copy-paste

```powershell
cd "c:\Users\user\Desktop\document reviewer"
pip install -r requirements.txt
python scripts/embed_rules.py --rebuild
python -m streamlit run streamlit_app.py
```
