
import os
import io
from typing import Tuple, Optional, Dict, List
import pandas as pd

from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

from store_map import STORE_TO_SUPERVISOR

_LAST_STORE_SCORE_DF: Optional[pd.DataFrame] = None

STORE_SYNONYMS = {"store", "store#", "store #", "store id", "storeid"}
SCORE_SYNONYMS = {"score", "csat", "csat score", "avg score", "average", "avg"}

def _norm(s: str) -> str:
    return str(s).strip().lower()

def _canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        cn = _norm(c)
        if cn in STORE_SYNONYMS:
            rename_map[c] = "Store"
        elif cn in SCORE_SYNONYMS:
            rename_map[c] = "Score"
    if rename_map:
        df = df.rename(columns=rename_map)
    df.columns = [str(c).strip() for c in df.columns]
    return df

def _has_required_columns(cols) -> bool:
    cset = {str(c).strip() for c in cols}
    return "Store" in cset and "Score" in cset

def _read_excel_byteslike(bytes_data: bytes, header=0) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(bytes_data), header=header)

def _load_store_score_with_sniffing(file) -> pd.DataFrame:
    if hasattr(file, "read"):
        raw = file.read()
    else:
        with open(file, "rb") as f:
            raw = f.read()
    df = _read_excel_byteslike(raw, header=0)
    df = _canonicalize_columns(df)
    if _has_required_columns(df.columns):
        return df
    tmp = _read_excel_byteslike(raw, header=None)
    for hdr_row in range(0, min(10, len(tmp))):
        row_vals = [_norm(x) for x in list(tmp.iloc[hdr_row].values)]
        row_set = set(row_vals)
        if any(x in row_set for x in STORE_SYNONYMS) and any(x in row_set for x in SCORE_SYNONYMS):
            df2 = _read_excel_byteslike(raw, header=hdr_row)
            df2 = _canonicalize_columns(df2)
            if _has_required_columns(df2.columns):
                return df2
    raise ValueError("Couldn't find 'Store' and 'Score' columns in the Store-Score sheet. Ensure a single header row.")

def parse_input_excels(store_score_file, reviews_file, ignore_unknown: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame, list]:
    if not store_score_file or not reviews_file:
        raise ValueError("Both Store-Score and Reviews files are required.")
    store_score_df = _load_store_score_with_sniffing(store_score_file)
    store_score_df["Store"] = store_score_df["Store"].astype(str).str.strip()
    store_score_df["Score"] = pd.to_numeric(store_score_df["Score"], errors="coerce")
    store_score_df = store_score_df.dropna(subset=["Store", "Score"])
    map_series = pd.Series(STORE_TO_SUPERVISOR, name="Supervisor")
    map_series.index = map_series.index.map(lambda x: str(x).strip())
    store_score_df = store_score_df.merge(
        map_series.rename_axis("Store").reset_index(),
        on="Store",
        how="left"
    )
    unknown = store_score_df[store_score_df["Supervisor"].isna()]["Store"].unique().tolist()
    if unknown and not ignore_unknown:
        raise ValueError(f"The following stores are missing from STORE_TO_SUPERVISOR mapping: {unknown}")
    if unknown and ignore_unknown:
        store_score_df = store_score_df[store_score_df["Supervisor"].notna()].copy()
    if hasattr(reviews_file, "read"):
        rb = reviews_file.read()
        reviews_df = pd.read_excel(io.BytesIO(rb))
    else:
        reviews_df = pd.read_excel(reviews_file)
    reviews_df.columns = [str(c).strip() for c in reviews_df.columns]
    global _LAST_STORE_SCORE_DF
    _LAST_STORE_SCORE_DF = store_score_df.copy()
    return store_score_df, reviews_df, unknown

def compute_supervisor_stats(store_score_df: pd.DataFrame) -> pd.DataFrame:
    df = store_score_df.copy()
    highs = (
        df.sort_values(["Supervisor", "Score"], ascending=[True, False])
          .groupby("Supervisor", as_index=False)
          .agg(**{"Highest Store": ("Store", "first"), "Highest CSAT": ("Score", "first")})
    )
    lows = (
        df.sort_values(["Supervisor", "Score"], ascending=[True, True])
          .groupby("Supervisor", as_index=False)
          .agg(**{"Lowest Store": ("Store", "first"), "Lowest CSAT": ("Score", "first")})
    )
    avgs_counts = (
        df.groupby("Supervisor", as_index=False)
          .agg(**{"Area Average": ("Score", "mean"), "Num Stores": ("Store", "count")})
    )
    out = highs.merge(lows, on="Supervisor").merge(avgs_counts, on="Supervisor")
    cols = ["Supervisor", "Highest Store", "Highest CSAT", "Lowest Store", "Lowest CSAT", "Area Average", "Num Stores"]
    out = out[cols].sort_values("Supervisor").reset_index(drop=True)
    for c in ["Highest CSAT", "Lowest CSAT", "Area Average"]:
        out[c] = out[c].astype(float).round(2)
    return out

def _build_store_reviews_map(reviews_df: Optional[pd.DataFrame]) -> Dict[str, List[str]]:
    store_reviews: Dict[str, List[str]] = {}
    if reviews_df is None or reviews_df.empty:
        return store_reviews
    df = reviews_df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    store_col = col_map.get("store") or next((c for c in df.columns if c.lower() in ["store id", "store#", "store #", "location"]), None)
    comments_col = col_map.get("comments") or next((c for c in df.columns if c.lower() in ["comment", "review", "review text"]), None)
    if not store_col or not comments_col:
        return store_reviews
    df[store_col] = df[store_col].astype(str).str.strip()
    df[comments_col] = df[comments_col].astype(str).fillna("")
    for store, grp in df.groupby(store_col):
        comments = [str(x).strip() for x in grp[comments_col].tolist() if str(x).strip() and str(x).strip().lower() not in {"nan","none","null","n/a","na"}]
        store_reviews[str(store)] = comments
    return store_reviews

def _select_low_score_stores(store_score_df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    df = store_score_df.copy()
    df["Store"] = df["Store"].astype(str).str.strip()
    return df[df["Score"] < float(threshold)][["Supervisor", "Store", "Score"]]

def _pick_up_to_n_stores_per_supervisor(low_df: pd.DataFrame, n: int) -> Dict[str, List[str]]:
    picks: Dict[str, List[str]] = {}
    if low_df is None or low_df.empty:
        return picks
    ordered = low_df.sort_values(["Supervisor", "Score"], ascending=[True, True])
    for sup, grp in ordered.groupby("Supervisor"):
        stores = grp["Store"].astype(str).tolist()
        picks[str(sup)] = stores[:max(0, int(n))]
    return picks

def _build_llm_digest_for_supervisor(sup: str, store_ids: List[str], store_reviews: Dict[str, List[str]], max_chars_per_store: int = 1200) -> str:
    lines = [f"Supervisor: {sup}"]
    for sid in store_ids:
        comments = store_reviews.get(str(sid), [])
        if not comments:
            continue
        blob = "\n".join([f"- {c}" for c in comments])
        if len(blob) > max_chars_per_store:
            blob = blob[:max_chars_per_store - 3] + "..."
        lines.append(f"Store {sid} reviews:")
        lines.append(blob)
    return "\n".join(lines).strip()

SYS_PROMPT = "You are a concise, plain-English report writer. Keep outputs tight, readable, and skimmable for busy managers. Avoid fluff. Use short sentences and bullets."

def _openai_client():
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK not available. Install openai>=1.0.")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI()

def generate_report_markdown(
    stats_df: pd.DataFrame,
    model: str = "gpt-4o-mini",
    temperature: float = 1.0,
    max_output_tokens: int = 900,
    reviews_df: Optional[pd.DataFrame] = None,
    low_score_threshold: float = 7.50,
    max_stores_per_supervisor: int = 3,
) -> str:
    global _LAST_STORE_SCORE_DF
    base_df = _LAST_STORE_SCORE_DF if _LAST_STORE_SCORE_DF is not None else stats_df.rename(columns={"Highest Store": "Store", "Highest CSAT": "Score"})
    store_reviews = _build_store_reviews_map(reviews_df)
    low_df = _select_low_score_stores(base_df, float(low_score_threshold))
    picks = _pick_up_to_n_stores_per_supervisor(low_df, int(max_stores_per_supervisor))

    sup_order = list(stats_df["Supervisor"].astype(str))
    digest_blocks = []
    for sup in sup_order:
        store_ids = picks.get(sup, [])
        digest = _build_llm_digest_for_supervisor(sup, store_ids, store_reviews)
        if digest:
            digest_blocks.append(digest)
        else:
            digest_blocks.append(f"Supervisor: {sup}\n( No low-scoring stores selected or no reviews found )")
    review_block = "\n\n".join(digest_blocks).strip()

    table_lines = ["Supervisor,Highest Store,Highest CSAT,Lowest Store,Lowest CSAT,Area Average,Num Stores"]
    for _, row in stats_df.iterrows():
        table_lines.append(
            f"{row['Supervisor']},{row['Highest Store']},{row['Highest CSAT']:.2f},"
            f"{row['Lowest Store']},{row['Lowest CSAT']:.2f},{row['Area Average']:.2f},{int(row['Num Stores'])}"
        )
    data_block = "\n".join(table_lines)

    md = ""
    try:
        client = _openai_client()
        prompt_lines = [
            "Write a grouped CSAT report in markdown. For EACH supervisor, produce a section in this exact order:",
            "0) Blank line for formatting purposes",           
            "1) H2 header with supervisor name (e.g., '## Alex')",
            "2) Three bullets: Highest store (and score), Lowest store (and score), Area average across N stores",
            "3) '### Suggestions' with 1–3 concise, actionable bullets derived from negative reviews",
            "4) '### Commendations' with 1–3 concise bullets derived from positive reviews",
            "5) '### Reviews' listing the selected low-scoring stores; under each store, bullet ALL provided review comments",
            "6) '### Wrap-up' — a medium-length paragraph (5-8 sentences) summarizing issues, strengths, and priorities.",
            "",
            "Stay concise. Use plain language. No implementation details about the generation process.",
            "",
            "NUMERIC DATA (per supervisor):",
            "```",
            data_block,
            "```",
            "REVIEW DIGESTS (analyze these fully):",
            "```",
            review_block,
            "```",
            f"Keep the total output under ~{max_output_tokens} tokens.",
        ]
        user_prompt = "\n".join(prompt_lines)

        resp = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_output_tokens,
        )
        if resp and getattr(resp, "choices", None):
            choice = resp.choices[0]
            if choice and hasattr(choice, "message") and choice.message and getattr(choice.message, "content", None):
                md = choice.message.content.strip()
    except Exception as e:
        md = f"**[LLM error]** {type(e).__name__}: {str(e)}"

    if not md or md.startswith("**[LLM error]**"):
        lines: List[str] = []
        sup_stats = {str(r["Supervisor"]): r for _, r in stats_df.iterrows()}
        for sup in sup_order:
            r = sup_stats[sup]
            lines.append(f"## {sup}")
            lines.append("")
            lines.append(f"- Highest: **{r['Highest Store']}** at **{float(r['Highest CSAT']):.2f}**")
            lines.append(f"- Lowest: **{r['Lowest Store']}** at **{float(r['Lowest CSAT']):.2f}**")
            lines.append(f"- Area average: **{float(r['Area Average']):.2f}** across **{int(r['Num Stores'])}** stores")
            lines.append("")
            lines.append("### Suggestions")
            if md.startswith("**[LLM error]**"):
                lines.append("- Narrative unavailable (LLM error above).")
            else:
                lines.append("- Narrative unavailable (no LLM output).")
            lines.append("")
            lines.append("### Commendations")
            if md.startswith("**[LLM error]**"):
                lines.append("- Narrative unavailable (LLM error above).")
            else:
                lines.append("- Narrative unavailable (no LLM output).")
            lines.append("")
            lines.append("### Reviews")
            store_ids = picks.get(sup, [])
            if store_ids:
                for sid in store_ids:
                    lines.append(f"- **Store {sid}**")
                    comments = store_reviews.get(str(sid), [])
                    if comments:
                        for c in comments:
                            if isinstance(c, str) and c.strip().lower() in {"nan","none","null","n/a","na"}:
                                continue
                            if isinstance(c, str) and not c.strip():
                                continue
                            lines.append(f"  - {c}")
                    else:
                        lines.append("  - (No comments found)")
            else:
                lines.append("- (No low-scoring stores selected or no reviews found)")
            lines.append("")
            lines.append("### Wrap-up")
            if md.startswith("**[LLM error]**"):
                lines.append("_Narrative unavailable (LLM error above)._")
            else:
                lines.append("_Narrative unavailable (no LLM output)._")
            lines.append("")
        md = "\n".join(lines).strip()

    return md

def render_pdf_from_markdown(markdown_text: str) -> bytes:
    text = (markdown_text or "").strip()
    if not text:
        text = "# CSAT Report\n(No content received)"
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=LETTER)
    width, height = LETTER
    left_margin = 0.75 * inch
    right_margin = 0.75 * inch
    top_margin = 0.75 * inch
    line_height = 12
    max_width = width - left_margin - right_margin
    y = height - top_margin
    for line in text.splitlines():
        if y < 1 * inch:
            c.showPage()
            y = height - top_margin
        t = c.beginText(left_margin, y)
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            title = line[level:].strip()
            size = 18 - (level - 1) * 2
            t.setFont("Helvetica-Bold", max(10, size))
            t.textLine(title)
            c.drawText(t)
            y -= line_height + 6
        elif line.startswith("- "):
            t.setFont("Helvetica", 11)
            t.textLine("• " + line[2:])
            c.drawText(t)
            y -= line_height
        else:
            t.setFont("Helvetica", 11)
            words = line.split()
            cur = ""
            for w in words:
                trial = (cur + " " + w).strip()
                if c.stringWidth(trial, "Helvetica", 11) > max_width:
                    t.textLine(cur)
                    c.drawText(t)
                    y -= line_height
                    t = c.beginText(left_margin, y)
                    t.setFont("Helvetica", 11)
                    cur = w
                else:
                    cur = trial
            if cur == "" and not words:
                y -= line_height
            else:
                t.textLine(cur)
                c.drawText(t)
                y -= line_height
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer.read()
