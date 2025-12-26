# fed_pressers/ingest/keyword_counts.py

import os
import pandas as pd
import re

from fed_pressers.config.keywords import FED_KEYWORDS, FED_SYNONYM_MAP
from core.ingest.loader import load_cleaned_transcript
from core.util.helpers import safe_kw


# ------------------------------------------------
# CORE STATS (unchanged from earnings)
# ------------------------------------------------

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


# ------------------------------------------------
# FED-SPECIFIC HELPERS
# ------------------------------------------------

def extract_quarter_from_filename(filename):
    """
    Expected filename:
      01-29-2025-fedPC.cleaned.txt
    """
    return filename.split("-fedPC")[0]


# ------------------------------------------------
# MAIN ENTRYPOINT
# ------------------------------------------------

def build_fed_keyword_counts(paths):
    rows = []

    for path in paths:
        cleaned = load_cleaned_transcript(path)
        stats, total_words = keyword_stats(cleaned, FED_KEYWORDS)

        fname = os.path.basename(path)
        row = {
            "file": fname,
            "total_words": total_words,
            "quarter": extract_quarter_from_filename(fname)
        }

        # raw keyword stats
        for kw in FED_KEYWORDS:
            safe = safe_kw(kw)
            row[f"{safe}_count"]    = stats[kw]["count"]
            row[f"{safe}_firstpos"] = stats[kw]["first_pos"]
            row[f"{safe}_density"]  = stats[kw]["density"]

        rows.append(row)

    df = pd.DataFrame(rows)

    # ------------------------------------------------
    # Merge synonyms → canonical keywords
    # ------------------------------------------------
    for canonical, raws in FED_SYNONYM_MAP.items():
        safe_can = safe_kw(canonical)

        df[f"{safe_can}_count"] = df[[f"{safe_kw(kw)}_count" for kw in raws]].sum(axis=1)
        df[f"{safe_can}_firstpos"] = df[[f"{safe_kw(kw)}_firstpos" for kw in raws]].min(axis=1)
        df[f"{safe_can}_density"] = df[[f"{safe_kw(kw)}_density" for kw in raws]].sum(axis=1)

    return df
