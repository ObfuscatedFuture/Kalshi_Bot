# src/models/predict.py

import numpy as np
from projects.earnings_calls.config.keyword import CANONICAL_KEYWORDS
from core.models.train import get_feature_cols_for_keyword



def predict_next_quarter(models, df_features_long):
    """
    Predict the most recent quarter for every keyword.
    Ensures NaN/inf cleanup before feeding features into model.
    Returns dict: {keyword: probability}
    """
    predictions = {}

    for kw in CANONICAL_KEYWORDS:
        kw_df = df_features_long[df_features_long["keyword"] == kw].reset_index(drop=True)

        # Last row (latest quarter)
        X_next = kw_df[get_feature_cols_for_keyword(kw)].iloc[-1:].values.astype(float)

        # 🔧 FIX: Remove NaN and infinite values before scaling/model
        X_next = np.nan_to_num(X_next, nan=0.0, posinf=0.0, neginf=0.0)

        kind = models[kw][0]

        if kind == "logistic":
            _, model, scaler = models[kw]
            X_scaled = scaler.transform(X_next)
            prob = model.predict_proba(X_scaled)[0][1]

        else:  # bayesian fallback
            prob = models[kw][1]

        predictions[kw] = float(prob)

    return predictions

