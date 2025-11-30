# src/models/bayes_hierarchical.py

import numpy as np
import pymc as pm
at = pm.math
from sklearn.preprocessing import StandardScaler

from config.keyword import CANONICAL_KEYWORDS
from src.util.helpers import safe_kw

# -------------------------------------------------------------
# FEATURE SET (FULL KEYWORD FEATURES, NO COMPETITOR COMPOSITES)
# -------------------------------------------------------------
# These are columns in df_long (wide_to_long output):
#   last4_any, last4_count, weight4,
#   last8_any, last8_count, weight8,
#   last10_any, last10_count, weight10,
#   firstpos, density,
#   trend (from f"{safe}_trend"),
#   eps_surprise_pct_target, sin_season, cos_season
BAYES_FEATURES = [
    "last4_any",
    "last4_count",
    "weight4",
    "last8_any",
    "last8_count",
    "weight8",
    "last10_any",
    "last10_count",
    "weight10",
    "firstpos",
    "density",
    "trend",
    "eps_surprise_pct_target",
    "sin_season",
    "cos_season",
]


# -------------------------------------------------------------
# PREPARE DATA FOR PYMC MODEL
# -------------------------------------------------------------
def _inject_trend_column(df):
    """
    Build a generic 'trend' column from the per-keyword trend columns
    like 'open_source_trend', 'europe_trend', etc.
    """
    df = df.copy()
    df["trend"] = 0.0

    for kw in CANONICAL_KEYWORDS:
        safe = safe_kw(kw)
        col = f"{safe}_trend"
        if col in df.columns:
            mask = df["keyword"] == kw
            df.loc[mask, "trend"] = df.loc[mask, col]

    return df


def build_design_matrix(df_long, feature_cols):
    """
    Returns:
        X: numpy array (N x F)
        y: numpy array (N,)
        keyword_idx: array of group indices for hierarchical pooling
        scaler: fitted scaler
    """
    # Add generic 'trend' column from the per-keyword trend cols
    df = _inject_trend_column(df_long)

    # Map keywords -> integer group indices
    keyword_to_idx = {kw: i for i, kw in enumerate(CANONICAL_KEYWORDS)}
    keyword_idx = df["keyword"].map(keyword_to_idx).values.astype("int32")

    # Ensure all required features exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0.0

    X_raw = df[feature_cols].values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    y = df["present"].values.astype("int32")

    return X, y, keyword_idx, scaler


# -------------------------------------------------------------
# TRAIN HIERARCHICAL BAYESIAN LOGISTIC REGRESSION
# -------------------------------------------------------------
def train_bayesian(df_long, mode="nuts_fast"):
    """
    mode = "advi"       -> extremely fast (seconds)
    mode = "nuts_fast"  -> development speed (~minutes)
    mode = "nuts_full"  -> heavy MCMC (use on a beefier box)

    Returns:
        model: PyMC model
        trace: posterior samples (InferenceData)
        scaler: fitted StandardScaler
    """
    feature_cols = BAYES_FEATURES

    X, y, keyword_idx, scaler = build_design_matrix(df_long, feature_cols)

    N, F = X.shape
    K = len(CANONICAL_KEYWORDS)

    with pm.Model() as model:

        global_p = y.mean()
        global_logit = np.log(global_p / (1 - global_p))

        mu_alpha = pm.Normal("mu_alpha", global_logit, 1.5)
        sigma_alpha = pm.HalfNormal("sigma_alpha", 1.0)
        alpha_raw = pm.Normal("alpha_raw", 0.0, 1.0, shape=K)
        alpha = pm.Deterministic("alpha", mu_alpha + sigma_alpha * alpha_raw)

        mu_beta = pm.Normal("mu_beta", 0.0, 1.0, shape=F)
        sigma_beta = pm.HalfNormal("sigma_beta", 1.0, shape=F)
        beta_raw = pm.Normal("beta_raw", 0.0, 1.0, shape=(K, F))
        beta = pm.Deterministic("beta", mu_beta + sigma_beta * beta_raw)


        # -----------------------------
        # LINEAR COMBINATION
        # -----------------------------
        logits = alpha[keyword_idx] + at.sum(beta[keyword_idx] * X, axis=1)

        # Likelihood
        pm.Bernoulli("obs", logit_p=logits, observed=y)

        # =============================
        # === TRAINING MODES ==========
        # =============================
        if mode == "advi":
            print("\n🔵 Running ADVI (variational inference)...")
            approx = pm.fit(
                n=20_000,
                method="advi",
                callbacks=[pm.callbacks.CheckParametersConvergence(tolerance=1e-3)],
            )
            trace = approx.sample(2000)

        elif mode == "nuts_fast":
            print("\n🟡 Running FAST NUTS (500 draw / 1000 tune / 2 chains)")
            trace = pm.sample(
                draws=500,
                tune=1000,
                chains=2,
                target_accept=0.95,
                max_treedepth=12,
                init="jitter+adapt_diag",
                nuts_sampler="numpyro",   # optional, faster & less divergent
            )

        elif mode == "nuts_full":
            print("\n🔴 Running FULL NUTS (2000 draw / 2000 tune / 4 chains)")
            trace = pm.sample(
                draws=2000,
                tune=2000,
                chains=4,
                target_accept=0.95,
                max_treedepth=12,
                init="jitter+adapt_diag",
                nuts_sampler="numpyro",   # optional
            )

        else:
            raise ValueError("Unknown mode: choose 'advi', 'nuts_fast', or 'nuts_full'")


    return model, trace, scaler


# -------------------------------------------------------------
# PREDICT NEXT QUARTER (POSTERIOR CIs)
# -------------------------------------------------------------
def predict_bayes_next_quarter(model, trace, scaler, df_future_long):
    """
    df_future_long must contain one row per keyword (the future quarter).
    Returns:
        dict: {keyword: {"mean": p, "ci_low": x, "ci_high": y}}
    """
    feature_cols = BAYES_FEATURES

    # Build generic 'trend' column first
    df_future = _inject_trend_column(df_future_long)

    # Prepare X
    X_raw = df_future[feature_cols].values.astype(float)
    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)
    X = scaler.transform(X_raw)

    # Keyword → index
    keyword_to_idx = {kw: i for i, kw in enumerate(CANONICAL_KEYWORDS)}
    kw_idx = df_future["keyword"].map(keyword_to_idx).values.astype("int32")

    # ---------------------------------------------------------
    # Extract posterior samples and reshape robustly
    # ---------------------------------------------------------
    # alpha_raw: xarray DataArray with dims like (chain, draw, K) or (K, chain, draw)
    # beta_raw:  xarray DataArray with dims like (chain, draw, K, F) or similar
    alpha_raw = trace.posterior["alpha"].stack(sample=("chain", "draw"))
    beta_raw = trace.posterior["beta"].stack(sample=("chain", "draw"))

    alpha_vals = alpha_raw.values  # unknown axis order, but size = K * S
    beta_vals = beta_raw.values    # unknown axis order, but size = K * F * S

    K = len(CANONICAL_KEYWORDS)
    F = len(feature_cols)

    # Infer sample size S from total size
    total_alpha = alpha_vals.size
    S = total_alpha // K
    alpha_s = alpha_vals.reshape(K, S)  # (K, S)

    total_beta = beta_vals.size
    S_beta = total_beta // (K * F)
    if S_beta != S:
        # In case alpha/beta stacked differently, recompute S from beta
        S = S_beta
        alpha_s = alpha_vals.reshape(K, S)
    beta_s = beta_vals.reshape(K, S, F)  # (K, S, F)

    results = {}

    for i, kw in enumerate(df_future["keyword"]):
        k = kw_idx[i]
        x = X[i]  # shape (F,)

        # alpha_s[k]: (S,)
        # beta_s[k]: (S, F)
        logits = alpha_s[k] + beta_s[k] @ x  # (S,)
        probs = 1.0 / (1.0 + np.exp(-logits))

        mean = probs.mean()
        lower = np.quantile(probs, 0.05)
        upper = np.quantile(probs, 0.95)

        results[kw] = {
            "mean": float(mean),
            "ci_low": float(lower),
            "ci_high": float(upper),
        }

    return results
