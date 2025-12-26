import pandas as pd
import numpy as np

# ============================
# CONFIG / INGEST
# ============================

# TODO: replace with your actual config module
from projects.fed_pressers.config.paths import fed_paths

# TODO: Fed-specific keyword counter
# Should accept: list[str] of .cleaned.txt paths
# Should return: DataFrame with columns:
#   - quarter
#   - keyword columns
from projects.fed_pressers.ingest.keyword_counts import build_fed_keyword_counts


# ============================
# FEATURES / MODELS
# ============================

# TODO: optional feature engineering (lags, smoothing, etc.)
from fed_pressers.features.make_feature_df import build_feature_frame

# Reuse your existing model code
from core.models.train import train_keyword_models
from core.models.predict import predict_next_quarter

# Optional helpers (can delete if unused)
from core.util.helpers import quarter_to_year_q


# ============================
# PIPELINE
# ============================

def run_fed_pipeline():
    # --------------------------------------------
    # 1. Build Fed keyword dataframe
    # --------------------------------------------
    df = build_fed_keyword_counts(fed_paths)

    # Expected minimal columns:
    # df["quarter"]
    # df["keyword_*"]

    # --------------------------------------------
    # 2. Time features (very lightweight)
    # --------------------------------------------
    df["quarter_order"] = df["quarter"].apply(quarter_to_year_q)
    df = df.sort_values("quarter_order").reset_index(drop=True)

    df["t"] = np.arange(len(df))
    df["sin_season"] = np.sin(2 * np.pi * df["t"] / 4)
    df["cos_season"] = np.cos(2 * np.pi * df["t"] / 4)

    # --------------------------------------------
    # 3. Feature engineering
    # --------------------------------------------
    df_features = build_feature_frame(df)

    # Optional persistence
    df_features.to_csv("fed_keyword_feature_wide.csv", index=False)

    # --------------------------------------------
    # 4. Wide → long
    # --------------------------------------------
    df_long = pd.melt(
        df_features,
        id_vars=["quarter", "t", "sin_season", "cos_season"],
        var_name="keyword",
        value_name="count"
    )

    df_long.to_csv("fed_keyword_feature_long.csv", index=False)

    # --------------------------------------------
    # 5. Train models
    # --------------------------------------------
    models = train_keyword_models(df_long)

    # --------------------------------------------
    # 6. Predict next meeting
    # --------------------------------------------
    preds = predict_next_quarter(models, df_long)

    # --------------------------------------------
    # 7. Display
    # --------------------------------------------
    print("\n=== NEXT FED MEETING KEYWORD PROBABILITIES ===\n")
    for kw, prob in sorted(preds.items(), key=lambda x: x[1], reverse=True):
        print(f"{kw:<25} | {prob * 100:6.2f}%")

    return preds


# ============================
# ENTRYPOINT
# ============================

if __name__ == "__main__":
    run_fed_pipeline()
