# src/models/train.py

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from config.keyword import CANONICAL_KEYWORDS
from config.companies import COMPETITORS, TARGET
from config.parameters import (
    LOGISTIC_C, PRIORS,
    MIN_POSITIVES, MIN_NEGATIVES
)
from src.util.helpers import safe_kw


def get_feature_cols_for_keyword(kw):
    """
    Build feature list for a given keyword:
    - target rolling & weighted features
    - competitor composites
    - seasonality + eps
    """
    safe = safe_kw(kw)
    base = [
        #"last4_any",
        "last4_count",
        #"weight4",
        #"last8_any",
        #"last8_count",
        #"weight8",
        #"last10_any",
        "last10_count",
        #"weight10",
        "firstpos",
        "density",
        #f"{safe}_trend",
        "eps_surprise_pct_target",
        "sin_season",
        "cos_season",
    ]

    competitor = [
        f"{safe}_{comp}_composite"
        for comp in COMPETITORS if comp != TARGET
    ]

    return base + competitor


def train_keyword_models(df_features_long):
    """
    Train logistic/Bayesian fallback model for each keyword.
    df_features_long is df_long (wide->long transformation).
    Returns dict: {kw: ("logistic", model, scaler) or ("bayes", p_hat)}
    """
    models = {}
    priors = PRIORS

    for kw in CANONICAL_KEYWORDS:
        safe = safe_kw(kw)

        kw_df = df_features_long[df_features_long["keyword"] == kw].reset_index(drop=True)

        y = kw_df["present"].values
        cols = get_feature_cols_for_keyword(kw)


        # ensure missing columns exist filled with zeros
        for c in cols:
            if c not in kw_df.columns:
                kw_df[c] = 0.0

        X = kw_df[cols].values.astype(float)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


        positives = int(y.sum())
        negatives = int((y == 0).sum())
        N = len(y)

        # logistic regression only if enough data
        if positives >= MIN_POSITIVES and negatives >= MIN_NEGATIVES:

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = LogisticRegression(C=LOGISTIC_C)
            model.fit(X_scaled, y)

            models[kw] = ("logistic", model, scaler)

        else:
            # Bayesian fallback probability
            alpha = priors["alpha"]
            beta = priors["beta"]
            p_hat = (positives + alpha) / (N + alpha + beta)
            models[kw] = ("bayes", p_hat)

    return models
