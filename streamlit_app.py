"""Streamlit UI for GDP document compliance review."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.compliance_pipeline import run_compliance_pipeline
from src.logging_config import setup_logging
from src.output_utils import run_timestamp

setup_logging()

UPLOAD_DIR = PROJECT_ROOT / "output" / "uploads"
DEFAULT_RULES_PATH = PROJECT_ROOT / "rules" / "rules.json"

st.set_page_config(
    page_title="Document Compliance Reviewer",
    page_icon="📄",
    layout="wide",
)

if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None


def save_upload(uploaded_file, run_ts: str) -> Path:
    folder = UPLOAD_DIR / run_ts
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / uploaded_file.name
    path.write_bytes(uploaded_file.getbuffer())
    return path


def status_color(status: str) -> str:
    return {
        "passed": "✅",
        "failed": "❌",
        "not_applicable": "⚪",
        "insufficient_evidence": "⚠️",
        "compliant": "✅",
        "non_compliant": "❌",
        "needs_review": "⚠️",
    }.get(status, "ℹ️")


def render_results(result: dict) -> None:
    if not result.get("success"):
        st.error("Compliance check did not complete successfully.")
        for error in result.get("errors", []):
            st.warning(error)
        if result.get("validation"):
            with st.expander("Validation details"):
                st.json(result["validation"])
        return

    report = result["report"]
    summary = report["summary"]
    validation = result["validation"]
    extracted = result["extracted"]
    chunks = result.get("chunks") or {}
    chunk_matches = result.get("chunk_rule_matches")

    overall = summary["overall_status"]
    st.subheader(f"{status_color(overall)} Overall status: {overall.replace('_', ' ').title()}")

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Passed", summary["passed"])
    col2.metric("Failed", summary["failed"])
    col3.metric("Not applicable", summary["not_applicable"])
    col4.metric("Needs review", summary["insufficient_evidence"])
    col5.metric("Total rules", summary["total_rules"])

    st.markdown("### Run details")
    details_col1, details_col2, details_col3 = st.columns(3)
    details_col1.write(f"**Run ID:** `{result['run_timestamp']}`")
    details_col2.write(f"**Extraction:** {report.get('extraction_mode', 'native')}")
    details_col3.write(f"**LLM engine:** {report.get('llm_engine', 'dummy')}")
    details_col1.write(f"**Pages:** {extracted.get('page_count', 0)}")
    details_col2.write(f"**Chunks:** {chunks.get('chunk_count', 0) if chunks else '— (whole doc)'}")
    details_col3.write(
        f"**Compliance mode:** {report.get('mode', 'whole_doc')} "
        f"({report.get('rule_retrieval', 'all')} rules)"
    )

    if chunk_matches:
        with st.expander("Chunk → matched rules (RAG)"):
            for match in chunk_matches.get("matches", []):
                st.markdown(
                    f"**{match['chunk_id']}** ({match.get('heading', '')}): "
                    f"{', '.join(match.get('matched_rule_ids', []))}"
                )

    rows = []
    for item in report.get("results", []):
        rows.append(
            {
                "Rule ID": item.get("rule_id"),
                "Title": item.get("title"),
                "Status": item.get("status"),
                "Severity": item.get("severity"),
                "Confidence": item.get("confidence"),
                "Reason": item.get("reason"),
                "Evidence": item.get("evidence"),
            }
        )

    if rows:
        st.markdown("### Rule results")
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        failed = [row for row in rows if row["Status"] == "failed"]
        if failed:
            st.markdown("### Failed rules")
            for row in failed:
                st.error(f"{row['Rule ID']} — {row['Title']}: {row['Reason']}")

    with st.expander("Validation summary"):
        st.json(validation)

    with st.expander("Compliance report JSON"):
        st.json(report)

    report_json = json.dumps(report, indent=2, ensure_ascii=False)
    st.download_button(
        "Download report JSON",
        data=report_json,
        file_name=f"compliance_report_{result['run_timestamp']}.json",
        mime="application/json",
    )


st.title("📄 Document Compliance Reviewer")
st.caption("GDP validation → extract → (chunk + RAG) → compliance check")

tab_upload, tab_results = st.tabs(["Upload & Check", "Results & Metrics"])

with tab_upload:
    st.subheader("Upload documents")
    doc_file = st.file_uploader(
        "Upload document",
        type=["pdf", "docx", "txt"],
        help="Supported formats: PDF, DOCX, TXT",
    )
    rules_file = st.file_uploader(
        "Upload rules file (optional)",
        type=["json", "txt"],
        help="Optional. If omitted, default rules/rules.json is used.",
    )

    st.subheader("Engine settings")
    setting_col1, setting_col2, setting_col3 = st.columns(3)
    extraction_mode = setting_col1.selectbox(
        "Extraction engine",
        options=["native", "ocr"],
        format_func=lambda x: "Native (pdfplumber/docx/txt)" if x == "native" else "OCR (EasyOCR, PDF only)",
    )
    compliance_mode = setting_col2.selectbox(
        "Compliance check mode",
        options=["whole_doc", "chunk_rag"],
        format_func=lambda x: (
            "Whole document (all 13 rules → LLM)"
            if x == "whole_doc"
            else "By chunk + RAG (vector match → dummy/real LLM)"
        ),
    )
    llm_engine = setting_col3.selectbox(
        "LLM engine",
        options=["dummy", "model_garden"],
        format_func=lambda x: "Dummy (local test)" if x == "dummy" else "Model Garden (real LLM)",
    )

    if compliance_mode == "chunk_rag":
        st.info("Chunk + RAG mode requires embedded rules. Run `python scripts/embed_rules.py` once first.")

    if extraction_mode == "ocr":
        st.info("OCR mode supports PDF only and may take longer on first run.")

    if llm_engine == "model_garden":
        st.warning("Model Garden requires `.env` credentials on this machine.")

    run_button = st.button("Run Compliance Check", type="primary", use_container_width=True)

    if run_button:
        if not doc_file:
            st.error("Please upload a document before running the compliance check.")
        else:
            with st.spinner("Running validation, extraction, chunking, and compliance check..."):
                try:
                    run_ts = run_timestamp()
                    doc_path = save_upload(doc_file, run_ts)
                    rules_path = None
                    if rules_file:
                        rules_path = save_upload(rules_file, run_ts)
                        if rules_path.suffix.lower() == ".txt":
                            st.warning("Plain-text rules upload is not wired yet. Using default rules/rules.json.")
                            rules_path = None

                    result = run_compliance_pipeline(
                        doc_path,
                        extraction_mode=extraction_mode,
                        compliance_mode=compliance_mode,
                        rules_path=rules_path,
                        use_dummy_llm=(llm_engine == "dummy"),
                        run_ts=run_ts,
                        #rules_path=Path("output/rules/generated_rules_20260630_112055.json")
                    )
                    st.session_state.pipeline_result = result

                    if result.get("success"):
                        st.success("Compliance check completed successfully. Open the Results tab.")
                    else:
                        st.error(f"Pipeline stopped at stage: {result.get('stage', 'unknown')}")
                except Exception as error:
                    st.error(f"Pipeline failed: {error}")

with tab_results:
    st.subheader("Results and metrics")
    if st.session_state.pipeline_result is None:
        st.info("No results yet. Run a compliance check from the Upload & Check tab.")
    else:
        render_results(st.session_state.pipeline_result)
