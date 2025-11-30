# src/ingest/keyword_counts.py

import os
import pandas as pd
import re
from config.keyword import RAW_KEYWORDS, SYNONYM_MAP
from src.ingest.loader import load_cleaned_transcript
from src.util.helpers import quarter_to_year_q
from src.util.helpers import safe_kw

def keyword_stats(text, keywords):
    words = text.split()
    total_words = len(words)

    stats = {}
    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r's?\b'
        matches = list(re.finditer(pattern, text))
        count = len(matches)
        first_pos = matches[0].start() / len(text) if count > 0 else 1.0
        density = (count / total_words) * 1000 if total_words else 0

        stats[kw] = {
            "count": count,
            "first_pos": first_pos,
            "density": density
        }

    return stats, total_words

def extract_quarter(filename):
    return filename.split("-")[0]

def build_company_counts(paths, eps_surprise=None):
    rows = []
    for path in paths:
        cleaned = load_cleaned_transcript(path)
        stats, total_words = keyword_stats(cleaned, RAW_KEYWORDS)

        row = {"file": os.path.basename(path), "total_words": total_words}
        row["quarter"] = extract_quarter(row["file"])

        for kw in RAW_KEYWORDS:
            safe = safe_kw(kw)
            row[f"{safe}_count"]    = stats[kw]["count"]
            row[f"{safe}_firstpos"] = stats[kw]["first_pos"]
            row[f"{safe}_density"]  = stats[kw]["density"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # merge synonyms → canonical columns
    for canonical, raws in SYNONYM_MAP.items():
        df[f"{safe_kw(canonical)}_count"] = df[[f"{safe_kw(kw)}_count" for kw in raws]].sum(axis=1)
        df[f"{safe_kw(canonical)}_firstpos"] = df[[f"{safe_kw(kw)}_firstpos" for kw in raws]].min(axis=1)
        df[f"{safe_kw(canonical)}_density"] = df[[f"{safe_kw(kw)}_density" for kw in raws]].sum(axis=1)

    if eps_surprise:
        df["eps_surprise_pct"] = df["quarter"].map(eps_surprise)

    return df
