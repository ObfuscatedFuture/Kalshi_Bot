# src/trends/fetch_trends.py

import os
import pandas as pd
from pytrends.request import TrendReq

from projects.earnings_calls.config.keyword import TRENDS_SYNONYMS, CANONICAL_KEYWORDS
from projects.earnings_calls.config.companies import REPORT_DATES, TARGET
from core.util.date_windows import get_trends_window
from core.util.helpers import safe_kw

# ---------------------------------------------
# PyTrends session
# ---------------------------------------------
pytrends = TrendReq(
    hl='en-US',
    tz=360,
    retries=5,
    backoff_factor=0.4
)

CACHE_DIR = "cache/trends"
COMPANY = TARGET.lower()


# ---------------------------------------------------------
# 1) Load or fetch multi-year history (cached)
# ---------------------------------------------------------
def load_or_fetch_full_term(term: str):
    """Load cached JSON or fetch once from Google Trends."""
    os.makedirs(CACHE_DIR, exist_ok=True)

    safe = term.replace(" ", "_")
    path = os.path.join(CACHE_DIR, f"{safe}_FULL.json")

    # ---------- CACHE HIT ----------
    if os.path.exists(path):
        df = pd.read_json(path, orient="split")
        df.index = pd.to_datetime(df.index, unit="ms")   # FIXED
        df.columns = [c.replace(" ", "_") + "_trend" for c in df.columns]  # FIXED
        return df

    # ---------- CACHE MISS ----------
    print(f"[FETCH] Google Trends for: {term}")
    timeframe = "2019-01-01 2025-12-31"

    try:
        pytrends.build_payload([term], timeframe=timeframe)
        df = pytrends.interest_over_time()
    except Exception as e:
        print(f"[ERROR] Fetch failed for {term}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"[WARN] Empty trends for {term}")
        return pd.DataFrame()

    if "isPartial" in df:
        df = df.drop(columns=["isPartial"])

    # Normalize index + columns before saving
    df.index = pd.to_datetime(df.index)
    df.columns = [c.replace(" ", "_") + "_trend" for c in df.columns]

    df.to_json(path, orient="split")
    return df


# ---------------------------------------------------------
# 2) Slice to quarterly window
# ---------------------------------------------------------
def slice_window(df, start, end):
    if df.empty:
        return df
    return df[(df.index >= start) & (df.index <= end)]


# ---------------------------------------------------------
# 3) Build trend feature for keyword × quarter
# ---------------------------------------------------------
def build_trends_for_keyword_quarter(kw, quarter, report_date_str):
    safe = safe_kw(kw)
    colname = f"trend_{COMPANY}_{safe}"

    start, end = get_trends_window(report_date_str)
    dfs = []

    for syn in TRENDS_SYNONYMS.get(kw, []):
        term = f"{COMPANY} {syn}"
        df_full = load_or_fetch_full_term(term)
        df_window = slice_window(df_full, start, end)

        if not df_window.empty:
            dfs.append(df_window)

    # No data → return zero
    if not dfs:
        return pd.DataFrame({"quarter": [quarter], [colname]: [0]})

    # Merge across synonyms
    merged = dfs[0]
    for d in dfs[1:]:
        merged = merged.merge(d, left_index=True, right_index=True, how="outer")

    trend_cols = [c for c in merged.columns if c.endswith("_trend")]

    value = merged[trend_cols].max(axis=1).mean()

    return pd.DataFrame({"quarter": [quarter], colname: [value]})


# ---------------------------------------------------------
# 4) Build entire trend dataset
# ---------------------------------------------------------
def build_all_trends_per_quarter():
    trend_rows = []
    report_map = REPORT_DATES[COMPANY]

    for quarter, date_str in report_map.items():
        row = {"quarter": quarter}

        # Build trend feature for *each keyword* but store in ONE ROW
        for kw in CANONICAL_KEYWORDS:
            df_kw_q = build_trends_for_keyword_quarter(kw, quarter, date_str)
            col = df_kw_q.columns[1]   # trend keyword column
            row[col] = df_kw_q[col].iloc[0]

        trend_rows.append(row)

    return pd.DataFrame(trend_rows)
