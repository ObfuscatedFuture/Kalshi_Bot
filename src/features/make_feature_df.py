# src/features/make_feature_df.py

import numpy as np
import pandas as pd

from config.keyword import CANONICAL_KEYWORDS
from config.companies import COMPETITORS, TARGET
from config.parameters import DECAY_W4, DECAY_W8, DECAY_W10

from src.util.helpers import exp_weights
from src.features.rolling_features import rolling_stats
from src.features.composite_features import competitor_composite
from src.util.helpers import safe_kw

def build_feature_frame(df):
    """
    Build the df_features table:
    - target features (present, last4, last8, last10, weighted)
    - per-competitor composite features
    - seasonal features
    """

    # Precompute weights
    w4  = exp_weights(4,  decay=DECAY_W4)
    w8  = exp_weights(8,  decay=DECAY_W8)
    w10 = exp_weights(10, decay=DECAY_W10)

    feature_rows = []

    for idx in range(len(df)):
        # ----------------------------------------------------------
        # Base row
        # ----------------------------------------------------------
        row_features = {
            "file": df.loc[idx, "file_target"],
            "quarter": df.loc[idx, "quarter"],
            "eps_surprise_pct_target": df.loc[idx, "eps_surprise_pct"],
            "sin_season": df.loc[idx, "sin_season"],
            "cos_season": df.loc[idx, "cos_season"],
        }

        # ----------------------------------------------------------
        # Per-keyword features
        # ----------------------------------------------------------
        for kw in CANONICAL_KEYWORDS:

            # ---------- TARGET SERIES ----------
            safe = safe_kw(kw)
            target_series = df[f"{safe}_count"].fillna(0)

            current_t = target_series.iloc[idx]

            row_features[f"{safe}_present"] = 1 if current_t > 0 else 0

            # win=4
            last4, last4_padded = rolling_stats(target_series, idx, 4)
            row_features[f"{safe}_last4_any"]   = 1 if last4.sum() > 0 else 0
            row_features[f"{safe}_last4_count"] = int(last4.sum())
            row_features[f"{safe}_weight4"]     = float(np.dot(w4, last4_padded))

            # win=8
            last8, last8_padded = rolling_stats(target_series, idx, 8)
            row_features[f"{safe}_last8_any"]   = 1 if last8.sum() > 0 else 0
            row_features[f"{safe}_last8_count"] = int(last8.sum())
            row_features[f"{safe}_weight8"]     = float(np.dot(w8, last8_padded))

            # win=10
            last10, last10_padded = rolling_stats(target_series, idx, 10)
            row_features[f"{safe}_last10_any"]   = 1 if last10.sum() > 0 else 0
            row_features[f"{safe}_last10_count"] = int(last10.sum())
            row_features[f"{safe}_weight10"]     = float(np.dot(w10, last10_padded))

            # firstpos + density
            row_features[f"{safe}_firstpos"] = df.loc[idx, f"{safe}_firstpos"]
            row_features[f"{safe}_density"]  = df.loc[idx, f"{safe}_density"]

            # ---------- TREND FEATURE ----------
            trend_val = df.loc[idx, f"trend_{TARGET.lower()}_{safe}"]
            row_features[f"{safe}_trend"] = trend_val



            # ---------- COMPETITOR COMPOSITES ----------
            for comp_name in COMPETITORS:
                if comp_name == TARGET:
                    continue

                col = f"{comp_name}__{safe}_count"
                if col not in df.columns:
                    # no competitor data for this kw
                    row_features[f"{safe}_{comp_name}_composite"] = 0.0
                    continue

                comp_series = df[col].fillna(0)

                # competitor rolling slices (unpadded needed for co_pressure)
                last4c, last4c_p = rolling_stats(comp_series, idx, 4)
                last8c, last8c_p = rolling_stats(comp_series, idx, 8)
                last10c, last10c_p = rolling_stats(comp_series, idx, 10)

                comp_value = competitor_composite(
                    last4c_p, last8c_p, last10c_p,
                    w4, w8, w10
                )


                row_features[f"{safe}_{comp_name}_composite"] = float(comp_value)

        # end keyword loop
        feature_rows.append(row_features)

    df_features = pd.DataFrame(feature_rows)
    return df_features
