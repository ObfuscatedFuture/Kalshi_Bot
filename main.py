# main.py

import pandas as pd

from config.companies import COMPANIES, TARGET, COMPETITORS
from config.keyword import CANONICAL_KEYWORDS
from src.ingest.keyword_counts import build_company_counts
from src.align.align_competitors import align_competitor_to_target
from src.features.make_feature_df import build_feature_frame
from src.models.train import train_keyword_models
from src.models.predict import predict_next_quarter
from src.models.evaluate import (
    evaluate_loocv,
    compute_all_forecast_cis,
    run_lofo
)
from src.util.helpers import quarter_to_year_q
import numpy as np
import math
from src.util.helpers import red, green, yellow
from src.ingest.keyword_counts import safe_kw
from src.trends.fetch_trends import build_all_trends_per_quarter



def run_pipeline():
    # --------------------------------------------
    # 1. Build target dataframe
    # --------------------------------------------
    target_cfg = COMPANIES[TARGET]
    df_target = build_company_counts(
        target_cfg["paths"],
        target_cfg["eps_surprise"]
    )

    df_target["report_date_target"] = df_target["quarter"]
    df_target = df_target.rename(columns={"file": "file_target"})

    df = df_target.copy()

    # --------------------------------------------
    # 2. Add competitors
    # --------------------------------------------
    for comp in COMPETITORS:
        if comp == TARGET:
            continue

        comp_cfg = COMPANIES[comp]
        df_comp = build_company_counts(
            comp_cfg["paths"],
            comp_cfg["eps_surprise"]
        )

        df_comp = align_competitor_to_target(
            comp, TARGET, df_comp, df_target
        )

        prefix = f"{comp}__"
        df_comp = df_comp.add_prefix(prefix)
        df_comp = df_comp.rename(columns={f"{prefix}quarter": "quarter"})

        df = pd.merge(df, df_comp, on="quarter", how="left")

  
    df_trends = build_all_trends_per_quarter()

    df = df.merge(df_trends, on="quarter", how="left")

    # --------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------
    df["quarter_order"] = df["quarter"].apply(quarter_to_year_q)
    df = df.sort_values("quarter_order").reset_index(drop=True)

    df["t"] = np.arange(len(df))
    df["sin_season"] = np.sin(2 * np.pi * df["t"] / 4)
    df["cos_season"] = np.cos(2 * np.pi * df["t"] / 4)


    

    df_features = build_feature_frame(df)
    df_features.to_csv("keyword_feature_wide.csv", index=False)

    # --------------------------------------------
    # 4. Wide → long (df_long)
    #    (the same as your original wide_to_long)
    # --------------------------------------------
    df_long = []
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
                f"{safe}_trend": row[f"{safe}_trend"]
            }

            # competitor composites
            for comp in COMPETITORS:
                if comp == TARGET:
                    continue
                entry[f"{safe}_{comp}_composite"] = row[f"{safe}_{comp}_composite"]

            df_long.append(entry)

    df_long = pd.DataFrame(df_long)
    df_long.to_csv("keyword_feature_long.csv", index=False)

    print(df_long.groupby("keyword")["present"].mean().sort_values())

    # --------------------------------------------
    # 5. Train models
    # --------------------------------------------
    models = train_keyword_models(df_long)

    # --------------------------------------------
    # 6. Predict next quarter
    # --------------------------------------------
    preds = predict_next_quarter(models, df_long)
    def pretty_print_predictions(preds):
        print("\n==================== MODEL PREDICTIONS ====================\n")
        print(f"{'KEYWORD':<20} | {'PROB (%)':>8}")
        print("-" * 40)

        for kw, prob in sorted(preds.items(), key=lambda x: x[1], reverse=True):
            pct = prob * 100

            if pct > 70:
                color = green
            elif pct > 30:
                color = yellow
            else:
                color = red

            print(f"{kw:<20} | {color(f'{pct:>6.2f}%')}")

        print("\n===========================================================\n")

    print("\n=== NEXT QUARTER PREDICTIONS ===")

    pretty_print_predictions(preds)

    # --------------------------------------------
    # 7. LOOCV evaluation
    # --------------------------------------------
    print("\n=== LOOCV RESULTS ===")
    eval_df = evaluate_loocv(df_long)
    print(eval_df)

    # --------------------------------------------
    # 8. Forecast confidence intervals
    # --------------------------------------------
    print("\n=== FORECAST CIs ===")
    ci_df = compute_all_forecast_cis(df_long, models, conf=0.50)
    print(ci_df)

    # --------------------------------------------
    # 9. LOFO feature importance
    # --------------------------------------------
    print("\n=== LOFO IMPORTANCE ===")
    print("See lofo_importance.csv for full results.")
    lofo_df = run_lofo(df_long)
    lofo_df.to_csv("lofo_importance.csv", index=False)

    return preds, eval_df, ci_df, lofo_df


if __name__ == "__main__":
    run_pipeline()
