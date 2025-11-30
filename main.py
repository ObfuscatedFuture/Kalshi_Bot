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
from src.util.helpers import red, green, yellow
from src.ingest.keyword_counts import safe_kw
from src.trends.fetch_trends import build_all_trends_per_quarter
from src.models.bayes_hierarchical import train_bayesian
from src.models.bayes_hierarchical import predict_bayes_next_quarter
from src.features.make_feature_df import build_future_base_row
from src.util.helpers import next_quarter_str
from src.features.wide_to_long import wide_to_long


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


    future_row = build_future_base_row(df)
    df2 = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)

    df_features = build_feature_frame(df)
    df_bhlr_features = build_feature_frame(df2)
    df_features.to_csv("keyword_feature_wide.csv", index=False)

    # --------------------------------------------
    # 4. Wide → long (df_long)
    # --------------------------------------------
    
    df_long = wide_to_long(df_features)
    df_long2 = wide_to_long(df_bhlr_features)
    df_long2.to_csv("keyword_feature_long.csv", index=False)

    # --------------------------------------------
    # 5. Train models
    # --------------------------------------------
    models = train_keyword_models(df_long)

    bayes_model, bayes_trace, bayes_scaler = train_bayesian(df_long, mode="nuts_fast")

    future_q = df_bhlr_features["quarter"].iloc[-1]
    df_next_long = df_long2[df_long2["quarter"] == future_q].reset_index(drop=True)


    # --------------------------------------------
    # 6. Predict next quarter
    # --------------------------------------------
    preds = predict_next_quarter(models, df_long)

    bayes_preds = predict_bayes_next_quarter(
        bayes_model,
        bayes_trace,
        bayes_scaler,
        df_next_long
    )
    
    def pretty_print_predictions(preds):
        print("\n==================== LOGISTIC REGRESSION PREDICTIONS ====================\n")
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
    
    def pretty_print_bayes(preds):
        print("\n===== HIERARCHICAL BAYES PREDICTIONS =====\n")
        for kw, r in preds.items():
            mean = r["mean"] * 100
            lo = r["ci_low"] * 100
            hi = r["ci_high"] * 100
            print(f"{kw:<15}  {mean:6.2f}%   (CI: {lo:5.2f}% – {hi:5.2f}%)")
        print("\n==========================================\n")


    print("\n=== NEXT QUARTER PREDICTIONS ===")

    pretty_print_predictions(preds)

    pretty_print_bayes(bayes_preds)

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
