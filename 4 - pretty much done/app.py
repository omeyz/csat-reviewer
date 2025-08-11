
import os
import io
import time
import streamlit as st
import pandas as pd

from csat_core import (
    parse_input_excels,
    compute_supervisor_stats,
    generate_report_markdown,
    render_pdf_from_markdown,
)

st.set_page_config(page_title="CSAT Analyzer", page_icon="📊", layout="wide")

st.title("📊 CSAT Analyzer — Report Generator (LLM-only, Grouped)")
st.markdown("""
Upload both files:
1) **Store-Score** Excel with columns **Store** and **Score** (required; used for stats).
2) **Reviews** Excel (free-form; required; used for analysis).
""")

with st.sidebar:
    st.header("Settings")
    model = st.text_input("OpenAI Model", value="gpt-4o-mini")
    temperature = st.slider("Temperature", 0.0, 2.0, 1.0, 0.1)
    max_output_tokens = st.number_input("Max Output Tokens", min_value=256, max_value=8192, value=900, step=100)
    low_score_threshold = st.number_input("Low-score threshold", min_value=0.0, max_value=10.0, value=7.50, step=0.1)
    max_stores_per_supervisor = st.number_input("Max stores per supervisor (reviews list)", min_value=1, max_value=10, value=3, step=1)
    make_pdf = st.checkbox("Generate PDF", value=True)
    ignore_unknown = st.checkbox("Ignore unknown store IDs (drop them)", value=False)
    st.caption("Set OPENAI_API_KEY in your environment.")

col1, col2 = st.columns(2)

with col1:
    store_score_file = st.file_uploader("Store-Score Excel (.xlsx) — requires 'Store' and 'Score'", type=["xlsx"])
with col2:
    reviews_file = st.file_uploader("Reviews Excel (.xlsx) — required", type=["xlsx"])

if store_score_file and reviews_file:
    try:
        with st.spinner("Reading and validating your files..."):
            store_score_df, reviews_df, unknown = parse_input_excels(store_score_file, reviews_file, ignore_unknown=ignore_unknown)

        with st.expander("Parsed columns & preview", expanded=False):
            st.write("**Store-Score columns:**", list(store_score_df.columns))
            st.dataframe(store_score_df.head(10), use_container_width=True)
            st.write("**Reviews columns:**", list(reviews_df.columns))
            st.dataframe(reviews_df.head(10), use_container_width=True)
            st.write("**Reviews loaded:**", len(reviews_df), "rows")
            try:
                from csat_core import _select_low_score_stores, _pick_up_to_n_stores_per_supervisor
                low_df = _select_low_score_stores(store_score_df, float(low_score_threshold))
                picks = _pick_up_to_n_stores_per_supervisor(low_df, int(max_stores_per_supervisor))
                st.write("**Low-scoring stores selected (per supervisor):**", picks)
            except Exception as _e:
                st.write("Diagnostics error:", str(_e))

        with st.spinner("Computing supervisor stats..."):
            stats_df = compute_supervisor_stats(store_score_df)

        st.success("Data parsed successfully ✅")
        st.dataframe(stats_df, use_container_width=True)

        with st.spinner("Generating report..."):
            report_md = generate_report_markdown(
                stats_df=stats_df,
                model=model,
                temperature=temperature,
                max_output_tokens=int(max_output_tokens),
                reviews_df=reviews_df,
                low_score_threshold=float(low_score_threshold),
                max_stores_per_supervisor=int(max_stores_per_supervisor),
            )

        st.subheader("Report Preview (Markdown)")
        st.markdown(report_md)

        # Diagnostics: surface LLM errors if present at the top of report
        if report_md.startswith("**[LLM error]**"):
            with st.expander("LLM diagnostics", expanded=True):
                st.error(report_md)

        st.caption(f"Report characters: {len(report_md)}")
        st.download_button(
            label="⬇️ Download Raw Markdown",
            data=report_md.encode("utf-8"),
            file_name=f"CSAT_Report_{int(time.time())}.md",
            mime="text/markdown",
        )

        if make_pdf:
            with st.spinner("Rendering PDF..."):
                pdf_bytes = render_pdf_from_markdown(report_md)

            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name=f"CSAT_Report_{int(time.time())}.pdf",
                mime="application/pdf",
            )
        else:
            st.info("PDF generation disabled in the sidebar.")

    except Exception as e:
        st.error(f"Something went wrong: {e}")
else:
    st.info("Upload both files to begin.")
