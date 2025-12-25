# src/models/evaluate.py

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss, log_loss

from projects.earnings_calls.config.keyword import CANONICAL_KEYWORDS
from projects.earnings_calls.config.parameters import (
    MIN_POSITIVES, MIN_NEGATIVES,
    LOGISTIC_C
)
from core.models.train import get_feature_cols_for_keyword


# ============================================================
# 1. LOOCV
# ============================================================
def loocv_metric(kw_df, cols):
    y_true = []
    y_pred = []

    for i in range(len(kw_df)):
        train = kw_df.drop(i)
        test = kw_df.iloc[i:i+1]

        y_train = train["present"].values
        y_test = test["present"].values

        X_train = train[cols].astype(float).values
        X_test  = test[cols].astype(float).values

        # 🔥 FIX: remove NaN + Inf
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

        positives = y_train.sum()
        negatives = (y_train == 0).sum()

        if positives >= MIN_POSITIVES and negatives >= MIN_NEGATIVES:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(C=LOGISTIC_C)
            model.fit(X_train_s, y_train)
            p = model.predict_proba(X_test_s)[0][1]

        else:
            p = (positives + 0.5) / (len(y_train) + 1.0)

        y_true.append(int(y_test[0]))
        y_pred.append(float(p))

    try:
        return log_loss(y_true, y_pred, labels=[0, 1])
    except:
        return float("inf")

def evaluate_loocv(df_features_long):
    """Return LOOCV statistics for all keywords."""
    rows = []

    for kw in CANONICAL_KEYWORDS:
        kw_df = df_features_long[df_features_long["keyword"] == kw].reset_index(drop=True)
        cols = get_feature_cols_for_keyword(kw)

        preds = []
        y_true = []

        for i in range(len(kw_df)):
            train = kw_df.drop(i).reset_index(drop=True)
            test = kw_df.iloc[i:i+1]

            y_train = train["present"].values
            X_train = train[cols].astype(float).values

            y_test = test["present"].values
            X_test = test[cols].astype(float).values

            # 🔥 FIX: Replace NaN/Inf in train & test matrices
            X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
            X_test  = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

            positives = y_train.sum()
            negatives = (y_train == 0).sum()

            if positives >= MIN_POSITIVES and negatives >= MIN_NEGATIVES:
                scaler = StandardScaler()
                X_train_s = scaler.fit_transform(X_train)
                X_test_s = scaler.transform(X_test)

                model = LogisticRegression(C=LOGISTIC_C)
                model.fit(X_train_s, y_train)
                p = model.predict_proba(X_test_s)[0][1]

            else:
                # Bayesian fallback
                p = (positives + 0.5) / (len(y_train) + 1.0)

            preds.append(float(p))
            y_true.append(int(y_test[0]))

        preds = np.array(preds)
        y_true = np.array(y_true)

        accuracy = (preds >= 0.5).astype(int).mean()
        brier = brier_score_loss(y_true, preds)

        try:
            ll = log_loss(y_true, preds, labels=[0, 1])
        except:
            ll = float("nan")

        rows.append({
            "keyword": kw,
            "accuracy": accuracy,
            "brier": brier,
            "log_loss": ll
        })

    return pd.DataFrame(rows)




# ============================================================
# 2. BOOTSTRAPPED FORECAST CONFIDENCE INTERVALS
# ============================================================

def bootstrap_forecast_ci(kw_df, model, feature_cols, n_boot=2000, conf=0.95):
    """
    Bootstrap CI for the most recent quarter.
    """
    # Extract all samples for this keyword
    X_all = kw_df[feature_cols].astype(float).values
    y_all = kw_df["present"].values

    # 🔥 FIX: sanitize the full matrix once
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)

    # Copy the newest row
    X_last = X_all[-1:].copy()
    n = len(y_all)

    preds = []

    for _ in range(n_boot):
        # Bootstrap sample indices
        idx = np.random.choice(np.arange(n), size=n, replace=True)

        Xb = X_all[idx]
        yb = y_all[idx]

        # 🔥 FIX: ensure no NaNs reappear
        Xb = np.nan_to_num(Xb, nan=0.0, posinf=0.0, neginf=0.0)

        positives = int(yb.sum())
        negatives = int((yb == 0).sum())

        # logistic or fallback
        if positives >= MIN_POSITIVES and negatives >= MIN_NEGATIVES:
            scaler = StandardScaler()
            Xb_s = scaler.fit_transform(Xb)
            Xlast_s = scaler.transform(X_last)

            lr = LogisticRegression(C=LOGISTIC_C)
            lr.fit(Xb_s, yb)
            p = lr.predict_proba(Xlast_s)[0][1]

        else:
            # Beta posterior mean (Bayesian fallback)
            p = (positives + 0.5) / (len(yb) + 1.0)

        preds.append(float(p))

    lo = float(np.percentile(preds, (1 - conf) / 2 * 100))
    hi = float(np.percentile(preds, (1 + conf) / 2 * 100))
    return lo, hi




def compute_all_forecast_cis(df_features_long, models, conf=0.95):
    rows = []

    for kw in CANONICAL_KEYWORDS:
        kw_df = df_features_long[df_features_long["keyword"] == kw].reset_index(drop=True)
        _, model_or_p, scaler = (models[kw] + (None,))[:3]  # tuple-safe

        cols = get_feature_cols_for_keyword(kw)

        lo, hi = bootstrap_forecast_ci(
            kw_df,
            models[kw],
            cols,
            conf=conf
        )

        rows.append({
            "keyword": kw,
            "ci_low": lo,
            "ci_high": hi
        })

    return pd.DataFrame(rows)



# ============================================================
# 3. LOFO — Leave-One-Feature-Out Importance
# ============================================================

def lofo_for_keyword(kw_df, kw):
    cols = get_feature_cols_for_keyword(kw)
    base = loocv_metric(kw_df, cols)

    rows = []
    for feat in cols:
        reduced = [c for c in cols if c != feat]
        score = loocv_metric(kw_df, reduced)
        rows.append({
            "keyword": kw,
            "feature": feat,
            "base_logloss": base,
            "drop_logloss": score,
            "delta_logloss": score - base,
        })

    return pd.DataFrame(rows)


def run_lofo(df_features_long):
    frames = []

    for kw in CANONICAL_KEYWORDS:
        kw_df = df_features_long[df_features_long["keyword"] == kw].reset_index(drop=True)
        try:
            frames.append(lofo_for_keyword(kw_df, kw))
        except Exception as e:
            print(f"[WARN] LOFO failed for {kw}: {e}")

    return pd.concat(frames, ignore_index=True)
