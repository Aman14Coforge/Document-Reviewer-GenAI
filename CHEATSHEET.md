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

Edit `.env` only for **Model Garden** (real LLM on office):

```env
MODEL_GARDEN_BASE_URL=https://your-gateway-url
MODEL_GARDEN_API_KEY=your_key
LLM_MODEL=your_model_name
```

---

## Streamlit UI

```powershell
python -m streamlit run streamlit_app.py
```

| Setting | Options |
|---------|---------|
| Extraction | `native` (digital PDF) · `ocr` (scanned PDF) |
| Compliance | `whole_doc` (all 13 rules) · `chunk_rag` (vector match) |
| LLM | `dummy` (no `.env`) · `model_garden` (needs `.env`) |

**Chunk + RAG requires embed first:** `python scripts/embed_rules.py --rebuild`

---

## Two compliance flows (CLI)

### Flow 1 — Whole document

Full text + **all 13 rules** → one LLM call.

```powershell
python scripts/run_whole_doc_check.py "docs/your_doc.pdf"
python scripts/run_whole_doc_check.py "docs/your_doc.pdf" --extraction ocr
python scripts/run_whole_doc_check.py "docs/your_doc.pdf" --llm model_garden
```

### Flow 2 — Chunk + RAG

Chunk → Chroma match → per-chunk LLM → merged report.

```powershell
python scripts/embed_rules.py --rebuild
python scripts/run_chunk_rag_check.py "docs/your_doc.pdf"
python scripts/run_chunk_rag_check.py "docs/your_doc.pdf" --llm model_garden
```

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

# 4. Match rules (RAG)
python scripts/match_chunk_rules.py "output/chunks/your_doc_<timestamp>_chunks.json"

# 5. Compliance (legacy CLI)
python scripts/check_compliance.py "output/chunks/..." --file-name "your_doc.pdf"
```

---

## Rules & RAG

| Task | Command |
|------|---------|
| Embed rules into Chroma | `python scripts/embed_rules.py --rebuild` |
| Upsert after editing `rules.json` | `python scripts/embed_rules.py` |
| Plain rules → JSON (Prompt 1) | `python scripts/rules_from_text.py` |

Chroma data: `data/chroma/` · Rules: `rules/rules.json`

---

## LLM quick pick

| Where | Local test | Real LLM |
|-------|------------|----------|
| Streamlit | LLM engine → **Dummy** | LLM engine → **Model Garden** |
| CLI | `--llm dummy` (default) | `--llm model_garden` |

Dummy = keyword heuristics, no API. Model Garden = needs `.env`.

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
| `data/chroma/` | Embedded rule vectors |

Filenames include a **timestamp** — runs never overwrite each other.

---

## Config files

| File | Change what |
|------|-------------|
| `config/validation.json` | Max size, allowed types, error messages |
| `config/compliance.json` | RAG top-k, always-include rules, prompt paths |
| `config/ocr.json` | EasyOCR languages, DPI |
| `.env` | Model Garden URL, key, model |

RAG defaults: top **5** rules per chunk + always **GDP-08, GDP-09, GDP-10**.

---

## Local vs office

| Component | Here (dev) | Office |
|-----------|------------|--------|
| LLM | Dummy or Model Garden | Model Garden |
| Embeddings | MiniLM + Chroma | Office API *(planned)* |

Re-run `embed_rules.py` on office if embedding model differs. Don't copy `data/chroma/` across different models.

---

## Report status meanings

| Status | Meaning |
|--------|---------|
| `compliant` | All rules passed or N/A |
| `non_compliant` | At least one rule failed |
| `needs_review` | No failures, but some rules lack evidence |

Per-rule: `passed` · `failed` · `not_applicable` · `insufficient_evidence`

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

---

## Sample doc

```powershell
python scripts/run_whole_doc_check.py "docs/Deployment Report_v0 - filled.pdf"
python scripts/run_chunk_rag_check.py "docs/Deployment Report_v0 - filled.pdf"
```

---

## One-liner copy-paste

```powershell
cd "c:\Users\user\Desktop\document reviewer"
pip install -r requirements.txt
python scripts/embed_rules.py --rebuild
python -m streamlit run streamlit_app.py
```
python -m venv .venv
.venv\Scripts\activate.bat - cmd
.venv\Scripts\Activate.ps1 - ps
python -m streamlit run streamlit_app.py