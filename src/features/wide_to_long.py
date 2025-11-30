# main.py

import pandas as pd

from config.companies import TARGET, COMPETITORS
from config.keyword import CANONICAL_KEYWORDS

from src.ingest.keyword_counts import safe_kw

def wide_to_long(df_features: pd.DataFrame) -> pd.DataFrame:
        """
        Convert wide SNOW feature table (one row per quarter)
        into long format (one row per quarter × keyword).
        """
        rows = []

        for _, row in df_features.iterrows():
            for kw in CANONICAL_KEYWORDS:
                safe = safe_kw(kw)

                entry = {
                    "file": row["file"],
                    "quarter": row["quarter"],
                    "keyword": kw,
                    "present": row[f"{safe}_present"],
                    "last4_any": row[f"{safe}_last4_any"],
                    "last4_count": row[f"{safe}_last4_count"],
                    "last8_any": row[f"{safe}_last8_any"],
                    "last8_count": row[f"{safe}_last8_count"],
                    "last10_any": row[f"{safe}_last10_any"],
                    "last10_count": row[f"{safe}_last10_count"],
                    "weight4": row[f"{safe}_weight4"],
                    "weight8": row[f"{safe}_weight8"],
                    "weight10": row[f"{safe}_weight10"],
                    "firstpos": row[f"{safe}_firstpos"],
                    "density": row[f"{safe}_density"],
                    "eps_surprise_pct_target": row["eps_surprise_pct_target"],
                    "sin_season": row["sin_season"],
                    "cos_season": row["cos_season"],
                    f"{safe}_trend": row[f"{safe}_trend"],
                }

                # competitor composites
                for comp in COMPETITORS:
                    if comp == TARGET:
                        continue
                    entry[f"{safe}_{comp}_composite"] = row[f"{safe}_{comp}_composite"]

                rows.append(entry)

        return pd.DataFrame(rows)