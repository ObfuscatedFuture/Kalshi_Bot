import pdfplumber
import re
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from datetime import datetime
import statsmodels.api as sm
from scipy.stats import chi2

# Config

# Transcript paths
hd_paths = [
    'dataset/HD-earnings/1Q23-transcript-HD.pdf',
    'dataset/HD-earnings/2Q23-transcript-HD.pdf',
    'dataset/HD-earnings/3Q23-transcript-HD.pdf',
    'dataset/HD-earnings/4Q23-transcript-HD.pdf',
    'dataset/HD-earnings/1Q24-transcript-HD.pdf',
    'dataset/HD-earnings/2Q24-transcript-HD.pdf',
    'dataset/HD-earnings/3Q24-transcript-HD.pdf',
    'dataset/HD-earnings/4Q24-transcript-HD.pdf',
    'dataset/HD-earnings/1Q25-transcript-HD.pdf',
    'dataset/HD-earnings/2Q25-transcript-HD.pdf',
    'dataset/HD-earnings/3Q25-transcript-HD.pdf',
]

lowes_paths = [
    'dataset/Lowe-earnings/1Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/2Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/3Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/4Q23-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/1Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/2Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/3Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/4Q24-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/1Q25-transcript-LOWE.pdf',
    'dataset/Lowe-earnings/2Q25-transcript-LOWE.pdf'
]

target_paths = [
    'dataset/Target-earnings/1Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/2Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/3Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/4Q23-transcript-TGT.pdf',
    'dataset/Target-earnings/1Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/2Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/3Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/4Q24-transcript-TGT.pdf',
    'dataset/Target-earnings/1Q25-transcript-TGT.pdf',
    'dataset/Target-earnings/2Q25-transcript-TGT.pdf'
]

walmart_paths = [
    'dataset/Walmart-earnings/1Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/2Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/3Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/4Q23-transcript-WMT.pdf',
    'dataset/Walmart-earnings/1Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/2Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/3Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/4Q24-transcript-WMT.pdf',
    'dataset/Walmart-earnings/1Q25-transcript-WMT.pdf',
    'dataset/Walmart-earnings/2Q25-transcript-WMT.pdf'
]

COMPETITORS = ["target", "lowes", "home_depot"]

# Full keyword list for raw counting (includes synonyms, caps preserved for PDF matching)
RAW_KEYWORDS = [
   "Tariff",
   "Tariffs",
   "Shutdown",
   "Shut down",
   "Brick and Mortar",
   "Dividend",
   "Dividends",
   "OnePay",
   "One Pay",
   "Omnichannel",
   "Prescription",
   "Prescriptions",
   "Competition",
   "Automation",
   "OpenAI",
   "Automotive",
   "Skip",
   "SNAP",
   "Upper Income"
]

# Canonical keyword list after merging synonyms
CANONICAL_KEYWORDS = [
   "tariff",
   "shutdown",
   "brick and mortar",
   "dividend",
   "onepay",
   "omnichannel",
   "prescription",
   "competition",
   "automation",
   "openai",
   "automotive",
   "skip",
   "snap",
   "upper income"
]

# Explicit mapping from canonical → raw variants
SYNONYM_MAP = {
    "tariff": ["Tariff", "Tariffs"],
    "shutdown": ["Shutdown", "Shut down"],
    "brick and mortar": ["Brick and Mortar"],
    "dividend": ["Dividend", "Dividends"],
    "onepay": ["OnePay", "One Pay"],
    "omnichannel": ["Omnichannel"],
    "prescription": ["Prescription", "Prescriptions"],
    "competition": ["Competition"],
    "automation": ["Automation"],
    "openai": ["OpenAI"],
    "automotive": ["Automotive"],
    "skip": ["Skip"],
    "snap": ["SNAP"],
    "upper income": ["Upper Income"]
}

REPORT_DATES = {
    "walmart": { # Done
        "1Q23": "2023-05-18",
        "2Q23": "2023-08-17",
        "3Q23": "2023-11-16",
        "4Q23": "2024-02-20",
        "1Q24": "2024-05-16",
        "2Q24": "2024-08-15",
        "3Q24": "2024-11-19",
        "4Q24": "2025-02-20",
        "1Q25": "2025-05-15",
        "2Q25": "2025-08-21"
    },
    "target": { # Done
        "1Q23": "2023-05-17",
        "2Q23": "2023-08-16",
        "3Q23": "2023-11-15",
        "4Q23": "2024-03-05",
        "1Q24": "2024-05-22",
        "2Q24": "2024-08-22",
        "3Q24": "2024-11-20",
        "4Q24": "2024-03-04",
        "1Q25": "2025-05-21",
        "2Q25": "2025-08-20"
    },
    "home_depot": { # Done
        "1Q23": "2023-05-16",
        "2Q23": "2023-08-15",
        "3Q23": "2023-11-14",
        "4Q23": "2024-02-20",
        "1Q24": "2024-05-14",
        "2Q24": "2024-08-13",
        "3Q24": "2024-11-12",
        "4Q24": "2025-02-25",
        "1Q25": "2025-05-20",
        "2Q25": "2025-08-19",
        "3Q25": "2025-11-18"
    },
    "lowes": { # Done
        "1Q23": "2023-05-23",
        "2Q23": "2023-08-22",
        "3Q23": "2023-11-21",
        "4Q23": "2024-02-27",
        "1Q24": "2024-05-21",
        "2Q24": "2024-08-20",
        "3Q24": "2024-11-19",
        "4Q24": "2025-02-26",
        "1Q25": "2025-05-21",
        "2Q25": "2025-08-20",
    }

}


# Walmart EPS surprise data (as decimals)
WALMART_EPS_SURPRISE = {
    "1Q23": 0.1136,
    "2Q23": 0.0888,
    "3Q23": 0.0000,
    "4Q23": 0.0909,
    "1Q24": 0.1538,
    "2Q24": 0.0308,
    "3Q24": 0.0943,
    "4Q24": 0.0154,
    "1Q25": 0.0702,
    "2Q25": -0.0685
}

HOME_DEPOT_EPS_SURPRISE = {
    "1Q23": -0.052,
    "2Q23": -0.0409,
    "3Q23": -0.0131,
    "4Q23": -0.0177,
    "1Q24": -0.0055,
    "2Q24": -0.0278,
    "3Q24": -0.0370,
    "4Q24": -0.0288,
    "1Q25": 0.0084,
    "2Q25": 0.0064,
    "3Q25": -0.0184
}

TARGET_EPS_SURPRISE = {
    "1Q23": 0.1752,
    "2Q23": 0.2766,
    "3Q23": 0.4189,
    "4Q23": 0.2365,
    "1Q24": 0.0098,
    "2Q24": 0.1898,
    "3Q24": -0.1921,
    "4Q24": 0.0711,
    "1Q25": -0.1975,
    "2Q25": -0.0191,
    "3Q25": 0.0114
}

LOWES_EPS_SURPRISE = {
    "1Q23": 0.0546,
    "2Q23": 0.0156,
    "3Q23": 0.0033,
    "4Q23": 0.0536,
    "1Q24": 0.0408,
    "2Q24": 0.0354,
    "3Q24": 0.0248,
    "4Q24": 0.0546,
    "1Q25": 0.0139,
    "2Q25": 0.0236,
    "3Q25": 0.0303
}

# Companies config: add more companies by extending this dict
COMPANIES = {
    "walmart": {
        "paths": walmart_paths,
        "eps_surprise": WALMART_EPS_SURPRISE
    },
    "lowes": {
        "paths": lowes_paths,
        "eps_surprise": None
    },
    "home_depot": {
        "paths": hd_paths,
        "eps_surprise": HOME_DEPOT_EPS_SURPRISE
    },
    "target": {
        "paths": target_paths,
        "eps_surprise": TARGET_EPS_SURPRISE
    }
}

TARGET = "walmart"
COMPETITORS = ["target", "lowes", "home_depot"]

# -------------------------------------------------------------------
# 1. HELPERS: PDF + TEXT + DATES
# -------------------------------------------------------------------

# -------------------------------------------------------------------
# 1. HELPERS: TEXT-ONLY PIPELINE (after pre-cleaning)
# -------------------------------------------------------------------

def load_cleaned_transcript(pdf_path):
    """
    Instead of reading PDF → read the .cleaned.txt transcript produced 
    by your analyst-removal script.
    """
    txt_path = pdf_path.replace(".pdf", ".cleaned.txt")

    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        raise FileNotFoundError(f"🔥 Missing cleaned transcript:\n{txt_path}\n"
                                "Run the cleaner first!")

    return clean_transcript(text)


def clean_transcript(text):
    # Only do normalization — NOT analyst cleaning anymore
    text = text.lower()
    text = text.replace("\n", " ")
    text = re.sub(r"-\s*\n", "", text)       # hyphenated line breaks (rare now)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ")
    return text.strip()


def keyword_stats(text, keywords):
    words = text.split()
    total_words = len(words)

    stats = {}

    for kw in keywords:
        pattern = r'\b' + re.escape(kw.lower()) + r's?\b'
        matches = list(re.finditer(pattern, text))

        count = len(matches)
        first_pos = (
            matches[0].start() / len(text)
            if count > 0
            else 1.0
        )

        density = (count / total_words) * 1000 if total_words > 0 else 0

        stats[kw] = {
            "count": count,
            "first_pos": first_pos,
            "density": density
        }

    return stats, total_words


def extract_quarter(filename):
    return filename.split("-")[0]

def quarter_to_year_q(qstr):
    """
    Convert quarter string like '1Q23' to sortable numeric key 20231
    """
    q = int(qstr[0])           # e.g. "1Q23" → quarter = 1
    yr = int(qstr[2:])         # e.g. "1Q23" → year = 23
    year_full = 2000 + yr      # convert "23" → 2023
    return year_full * 10 + q  # unique sortable number (e.g. 20231)

def exp_weights(n, decay):
    w = np.array([decay**i for i in range(n)][::-1])
    return w / w.sum()

def get_report_date(company, quarter):
    return datetime.fromisoformat(REPORT_DATES[company][quarter])

# -------------------------------------------------------------------
# 2. BUILD COUNTS FOR A COMPANY (GENERIC)
# -------------------------------------------------------------------

def build_company_counts(paths, eps_dict=None):
    rows = []
    for path in paths:
        cleaned = load_cleaned_transcript(path)

        stats, total_words = keyword_stats(cleaned, RAW_KEYWORDS)

        row = {"file": os.path.basename(path), "total_words": total_words}

        # Raw columns
        for kw in RAW_KEYWORDS:
            row[f"{kw}_count"] = stats[kw]["count"]
            row[f"{kw}_firstpos"] = stats[kw]["first_pos"]
            row[f"{kw}_density"] = stats[kw]["density"]

        row["quarter"] = extract_quarter(row["file"])
        rows.append(row)

    df = pd.DataFrame(rows)

    # Merge synonyms into canonical columns
    for canonical, raws in SYNONYM_MAP.items():
        count_cols    = [f"{kw}_count" for kw in raws]
        firstpos_cols = [f"{kw}_firstpos" for kw in raws]
        density_cols  = [f"{kw}_density" for kw in raws]

        df[f"{canonical}_count"] = df[count_cols].sum(axis=1)
        df[f"{canonical}_firstpos"] = df[firstpos_cols].min(axis=1)
        df[f"{canonical}_density"]  = df[density_cols].sum(axis=1)

    # Drop all raw keyword columns
    raw_cols_to_drop = []
    for kw in RAW_KEYWORDS:
        raw_cols_to_drop.extend([
            f"{kw}_count",
            f"{kw}_firstpos",
            f"{kw}_density"
        ])
    df = df.drop(columns=raw_cols_to_drop)

    # Attach EPS surprise if provided
    if eps_dict is not None:
        df["eps_surprise_pct"] = df["quarter"].map(eps_dict)

    return df

# -------------------------------------------------------------------
# 3. ALIGN COMPETITORS TO TARGET BY REPORT DATE
# -------------------------------------------------------------------

def align_competitor_to_target(comp_name, target_name, df_comp, df_target):
    """
    For each competitor quarter, map it to the NEXT target earnings date.
    This handles 'extra' competitor quarters cleanly.
    """
    # Build sorted list of (target_quarter, target_date)
    target_dates = []
    for q in df_target["quarter"].unique():
        d = get_report_date(target_name, q)
        target_dates.append((q, d))
    target_dates = sorted(target_dates, key=lambda x: x[1])

    # Competitor report dates
    df_comp["report_date"] = df_comp["quarter"].apply(
        lambda q: get_report_date(comp_name, q)
    )

    aligned_quarters = []
    for _, row in df_comp.iterrows():
        d_comp = row["report_date"]
        matched_q = None
        for q_t, d_t in target_dates:
            if d_comp < d_t:
                matched_q = q_t
                break
        aligned_quarters.append(matched_q)

    df_comp["aligned_quarter"] = aligned_quarters

    # Drop rows that don't map to any existing target quarter yet (future info)
    df_comp = df_comp[~df_comp["aligned_quarter"].isna()].copy()
    df_comp["quarter"] = df_comp["aligned_quarter"]
    df_comp = df_comp.drop(columns=["aligned_quarter", "report_date"])

    return df_comp

# -------------------------------------------------------------------
# 4. BUILD MERGED TARGET + MULTI-COMPETITOR DATAFRAME
# -------------------------------------------------------------------
def build_merged_dataframe(target_name, competitors):
    target_cfg = COMPANIES[target_name]

    # --------------------------------------------------------
    # Build Target Counts
    # --------------------------------------------------------
    df_target = build_company_counts(
        target_cfg["paths"],
        target_cfg.get("eps_surprise")
    )
    df_target["report_date_target"] = df_target["quarter"].apply(
        lambda q: get_report_date(target_name, q)
    )
    df_target = df_target.rename(columns={"file": "file_target"})

    # Start merged DF with just the target
    df = df_target.copy()

    # --------------------------------------------------------
    # Add all competitors
    # --------------------------------------------------------
    for comp_name in competitors:
        if comp_name == target_name:
            continue

        comp_cfg = COMPANIES[comp_name]

        # raw competitor stats
        df_comp = build_company_counts(
            comp_cfg["paths"],
            comp_cfg.get("eps_surprise")
        )

        # align competitor quarter → target quarter based on reporting date
        df_comp = align_competitor_to_target(
            comp_name, target_name, df_comp, df_target
        )

        # Rename columns to avoid collisions
        prefix = f"{comp_name}__"
        df_comp = df_comp.add_prefix(prefix)
        df_comp = df_comp.rename(columns={f"{prefix}quarter": "quarter"})

        # Merge competitor table into target
        df = pd.merge(df, df_comp, on="quarter", how="left")

    # Seasonality
    df["quarter_order"] = df["quarter"].apply(quarter_to_year_q)
    df = df.sort_values("quarter_order").reset_index(drop=True)

    df["t"] = np.arange(len(df))
    df["sin_season"] = np.sin(2 * np.pi * df["t"] / 4)
    df["cos_season"] = np.cos(2 * np.pi * df["t"] / 4)

    return df

df = build_merged_dataframe(TARGET, COMPETITORS)

# -------------------------------------------------------------------
# 5. FEATURE ENGINEERING (TARGET + PER-COMPETITOR COMPOSITES)
# -------------------------------------------------------------------

w4  = exp_weights(4,  decay=0.7)
w8  = exp_weights(8,  decay=0.75)
w10 = exp_weights(10, decay=0.8)

feature_rows = []
for idx in range(len(df)):
    row_features = {
        "file": df.loc[idx, "file_target"],
        "quarter": df.loc[idx, "quarter"],
        "eps_surprise_pct_target": df.loc[idx, "eps_surprise_pct"],
        "sin_season": df.loc[idx, "sin_season"],
        "cos_season": df.loc[idx, "cos_season"]
    }

    for kw in CANONICAL_KEYWORDS:
        # ===================== TARGET FEATURES =====================
        # Treat missing counts as 0
        target_series = df[f"{kw}_count"].fillna(0)

        current_target = target_series.iloc[idx]
        row_features[f"{kw}_present"] = 1 if current_target > 0 else 0

        last4_t  = target_series.iloc[max(0, idx-4): idx]
        last8_t  = target_series.iloc[max(0, idx-8): idx]
        last10_t = target_series.iloc[max(0, idx-10): idx]

        last4_t_p  = np.pad(last4_t.values,  (4  - len(last4_t),  0), constant_values=0)
        last8_t_p  = np.pad(last8_t.values,  (8  - len(last8_t),  0), constant_values=0)
        last10_t_p = np.pad(last10_t.values, (10 - len(last10_t), 0), constant_values=0)

        row_features[f"{kw}_last4_any"]    = 1 if last4_t.sum()  > 0 else 0
        row_features[f"{kw}_last4_count"]  = int(last4_t.sum())
        row_features[f"{kw}_last8_any"]    = 1 if last8_t.sum()  > 0 else 0
        row_features[f"{kw}_last8_count"]  = int(last8_t.sum())
        row_features[f"{kw}_last10_any"]   = 1 if last10_t.sum() > 0 else 0
        row_features[f"{kw}_last10_count"] = int(last10_t.sum())

        row_features[f"{kw}_firstpos"] = df.loc[idx, f"{kw}_firstpos"]
        row_features[f"{kw}_density"]  = df.loc[idx, f"{kw}_density"]

        row_features[f"{kw}_weight4"]  = float(np.dot(w4,  last4_t_p))
        row_features[f"{kw}_weight8"]  = float(np.dot(w8,  last8_t_p))
        row_features[f"{kw}_weight10"] = float(np.dot(w10, last10_t_p))

        # ===================== PER-COMPETITOR COMPOSITES =====================
        for comp_name in COMPETITORS:
            if comp_name == TARGET:
                continue

            comp_col = f"{comp_name}__{kw}_count"
            if comp_col not in df.columns:
                # no data for this competitor/keyword at all
                row_features[f"{kw}_{comp_name}_composite"] = 0.0
                continue

            # Treat missing competitor counts as 0
            comp_series = df[comp_col].fillna(0)

            last4_c  = comp_series.iloc[max(0, idx-4): idx]
            last8_c  = comp_series.iloc[max(0, idx-8): idx]
            last10_c = comp_series.iloc[max(0, idx-10): idx]

            last4_c_p  = np.pad(last4_c.values,  (4  - len(last4_c),  0), constant_values=0)
            last8_c_p  = np.pad(last8_c.values,  (8  - len(last8_c),  0), constant_values=0)
            last10_c_p = np.pad(last10_c.values, (10 - len(last10_c), 0), constant_values=0)

            w4_c  = np.dot(w4,  last4_c_p)
            w8_c  = np.dot(w8,  last8_c_p)
            w10_c = np.dot(w10, last10_c_p)

            co_pressure = (
                last4_c.sum() +
                0.5 * last8_c.sum() +
                0.2 * last10_c.sum()
            )

            row_features[f"{kw}_{comp_name}_composite"] = (
                0.6 * w4_c +
                0.3 * w8_c +
                0.1 * w10_c +
                0.2 * co_pressure
            )

    feature_rows.append(row_features)


df_features = pd.DataFrame(feature_rows)


# -------------------------------------------------------------------
# 6. WIDE -> LONG (PER KEYWORD ROWS)
# -------------------------------------------------------------------

def wide_to_long(df_features, keywords, competitors, target_name):
    rows = []
    for _, row in df_features.iterrows():
        file = row["file"]
        quarter = row["quarter"]
        for kw in keywords:
            r = {
                "file": file,
                "quarter": quarter,
                "keyword": kw,
                # Target features
                "present": row[f"{kw}_present"],
                "last4_any": row[f"{kw}_last4_any"],
                "last4_count": row[f"{kw}_last4_count"],
                "last8_any": row[f"{kw}_last8_any"],
                "last8_count": row[f"{kw}_last8_count"],
                "last10_any": row[f"{kw}_last10_any"],
                "last10_count": row[f"{kw}_last10_count"],
                "weight4": row[f"{kw}_weight4"],
                "weight8": row[f"{kw}_weight8"],
                "weight10": row[f"{kw}_weight10"],
                "firstpos": row[f"{kw}_firstpos"],
                "density": row[f"{kw}_density"],
                # Macro / seasonal
                "eps_surprise_pct_target": row["eps_surprise_pct_target"],
                "sin_season": row["sin_season"],
                "cos_season": row["cos_season"],
            }

            # Per-competitor composites (separate columns)
            for comp_name in competitors:
                if comp_name == target_name:
                    continue
                composite_key = f"{kw}_{comp_name}_composite"
                val = row.get(composite_key, 0.0)
                r[composite_key] = 0.0 if pd.isna(val) else val

            rows.append(r)

    return pd.DataFrame(rows)

df_long = wide_to_long(df_features, CANONICAL_KEYWORDS, COMPETITORS, TARGET)

df_long.to_csv("keyword_long.csv", index=False)

# -------------------------------------------------------------------
# 7. TRAIN PER-KEYWORD MODELS
# -------------------------------------------------------------------
keyword_datasets = { kw: df_long[df_long["keyword"] == kw].reset_index(drop=True) for kw in CANONICAL_KEYWORDS } 

models = {} 
priors = {"alpha": 0.5, "beta": 0.5} # Jeffreys prior 

base_feature_cols = [ # Target rolling features 
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
    # EPS surprise (target) 
    "eps_surprise_pct_target", # Seasonality 
    "sin_season", "cos_season" 
    ]
def get_feature_cols_for_keyword(kw):
    base = [ 
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
        "eps_surprise_pct_target", 
        "sin_season", 
        "cos_season", ] 
    competitor = [ f"{kw}_{comp_name}_composite" for comp_name in COMPETITORS if comp_name != TARGET ] 
    return base + competitor


keyword_datasets = {
    kw: df_long[df_long["keyword"] == kw].reset_index(drop=True)
    for kw in CANONICAL_KEYWORDS
}
models = {}
priors = {"alpha": 0.5, "beta": 0.5}

for kw, kw_df in keyword_datasets.items():

    feature_cols = get_feature_cols_for_keyword(kw)

    y = kw_df["present"].values
    X = kw_df[feature_cols].values

    positives = int(y.sum())
    negatives = int(
        (y == 0).sum())
    N = len(y)

    if positives >= 2 and negatives >= 2:
        # Fit scaler on THIS keyword's dataset
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LogisticRegression(C=0.01)
        model.fit(X_scaled, y)

        # Save both model + scaler
        models[kw] = ("logistic", model, scaler)

    else:
        # Bayesian fallback
        alpha = priors["alpha"]
        beta = priors["beta"]
        p_hat = (positives + alpha) / (N + alpha + beta)

        models[kw] = ("bayes", p_hat)

# -------------------------------------------------------------------
# 8. PREDICT NEXT (MOST RECENT) TARGET QUARTER
# -------------------------------------------------------------------

predictions = {}

for kw, kw_df in keyword_datasets.items():

    feature_cols = get_feature_cols_for_keyword(kw)
    X_next = kw_df[feature_cols].iloc[-1:].values

    kind = models[kw][0]

    if kind == "logistic":
        _, model, scaler = models[kw]
        X_next_scaled = scaler.transform(X_next)
        prob = model.predict_proba(X_next_scaled)[0][1]

    else:  # bayesian fallback
        prob = models[kw][1]

    predictions[kw] = float(prob)

# -------------------------------------------------------------------
# 9. PRETTY PRINT
# -------------------------------------------------------------------

def green(text): return f"\033[92m{text}\033[0m"
def yellow(text): return f"\033[93m{text}\033[0m"
def red(text): return f"\033[91m{text}\033[0m"

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

pretty_print_predictions(predictions)

# -------------------------------------------------------------------
# 10. LOOCV EVALUATION
# -------------------------------------------------------------------

def bootstrap_ci(values, n_boot=3000, ci=0.95):
    """
    Bootstraps a CI for an array of predicted probabilities.
    """
    N = len(values)
    boot_means = []

    for _ in range(n_boot):
        sample = np.random.choice(values, size=N, replace=True)
        boot_means.append(sample.mean())

    lower = np.percentile(boot_means, (1 - ci) * 50)
    upper = np.percentile(boot_means, 100 - (1 - ci) * 50)
    return float(lower), float(upper)

def evaluate_loocv_with_preds(df_long, keywords):
    results = []
    all_preds = {}

    for kw in keywords:
        kw_df = df_long[df_long["keyword"] == kw].reset_index(drop=True)

        # correct feature list
        feature_cols = get_feature_cols_for_keyword(kw)

        y_true = []
        y_pred = []

        for i in range(len(kw_df)):
            train = kw_df.drop(i)
            test = kw_df.iloc[i:i+1]

            y_train = train["present"].values
            X_train = train[feature_cols].values

            y_test = test["present"].values
            X_test = test[feature_cols].values

            positives = y_train.sum()
            negatives = (y_train == 0).sum()

            if positives >= 2 and negatives >= 2:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                model = LogisticRegression(C=0.01)
                model.fit(X_train_scaled, y_train)

                p = model.predict_proba(X_test_scaled)[0][1]

            else:
                p = (positives + 0.5) / (len(y_train) + 1.0)

            y_true.append(int(y_test[0]))
            y_pred.append(float(p))


        all_preds[kw] = np.array(y_pred)

        accuracy = ((np.array(y_pred) >= 0.5).astype(int) == np.array(y_true)).mean()
        brier = brier_score_loss(y_true, y_pred)
        try:
            ll = log_loss(y_true, y_pred, labels=[0, 1])
        except:
            ll = float("nan")

        results.append({
            "keyword": kw,
            "accuracy": accuracy,
            "brier": brier,
            "log_loss": ll
        })

    return pd.DataFrame(results), all_preds

# -------------------------------------------------------------------
# 11. FORECAST CI — BOOTSTRAPPED FOR NEXT QUARTER ONLY
# -------------------------------------------------------------------

def bootstrap_forecast_ci_for_keyword(kw, kw_df, n_boot=2500, conf=0.95):
    """
    Bootstrap CI for the predicted probability for the MOST RECENT quarter
    (the last row in kw_df).
    """
    feature_cols = get_feature_cols_for_keyword(kw)
    X_all = kw_df[feature_cols].values
    y_all = kw_df["present"].values
    X_last = X_all[-1:].copy()
    n = len(y_all)

    boot_preds = []

    for _ in range(n_boot):
        idx = np.random.choice(np.arange(n), size=n, replace=True)
        Xb = X_all[idx]
        yb = y_all[idx]

        positives = int(yb.sum())
        negatives = int((yb == 0).sum())

        if positives >= 2 and negatives >= 2:
            scaler = StandardScaler()
            Xb_scaled = scaler.fit_transform(Xb)
            X_last_scaled = scaler.transform(X_last)

            model = LogisticRegression(C=0.01)
            model.fit(Xb_scaled, yb)
            p = model.predict_proba(X_last_scaled)[0][1]
        else:
            alpha = 0.5
            beta_prior = 0.5
            p = (positives + alpha) / (len(yb) + alpha + beta_prior)

        boot_preds.append(p)

    lower, upper = np.percentile(
        boot_preds,
        [(1 - conf) / 2 * 100, (1 - (1 - conf) / 2) * 100]
    )

    return float(lower), float(upper)

def compute_all_forecast_cis(keyword_datasets, predictions, conf=0.95):
    rows = []
    for kw, kw_df in keyword_datasets.items():
        ci_low, ci_high = bootstrap_forecast_ci_for_keyword(
            kw, kw_df, conf=conf
        )
        rows.append({
            "keyword": kw,
            "prediction": predictions[kw],
            "ci_low": ci_low,
            "ci_high": ci_high
        })
    return pd.DataFrame(rows)

df_ci = compute_all_forecast_cis(
    keyword_datasets,
    predictions,
    conf=0.50
)

print("\n==================== FORECAST CONFIDENCE INTERVALS ====================\n")
print(df_ci)
print("\n===========================================================================\n")


def lofo_importance_for_keyword(kw, kw_df):
    feature_cols = get_feature_cols_for_keyword(kw)
    y = kw_df["present"].values
    X = kw_df[feature_cols].values

    results = []

    # reference model performance (using LOOCV)
    base_perf = loocv_metric(kw_df, feature_cols)

    for feat in feature_cols:
        reduced_cols = [c for c in feature_cols if c != feat]
        reduced_perf = loocv_metric(kw_df, reduced_cols)

        results.append({
            "keyword": kw,
            "feature": feat,
            "base_logloss": base_perf,
            "drop_logloss": reduced_perf,
            "delta_logloss": reduced_perf - base_perf,
        })

    df = pd.DataFrame(results)
    return df.sort_values("delta_logloss", ascending=True)


def loocv_metric(kw_df, cols):
    """LOOCV log-loss for specified feature list."""
    y_true = []
    y_pred = []

    for i in range(len(kw_df)):
        train = kw_df.drop(i)
        test = kw_df.iloc[i:i+1]

        y_train = train["present"].values
        X_train = train[cols].values
        y_test = test["present"].values
        X_test = test[cols].values

        positives = y_train.sum()
        negatives = (y_train == 0).sum()

        if positives >= 2 and negatives >= 2:
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_test_s = scaler.transform(X_test)

            model = LogisticRegression(C=0.01)
            model.fit(X_train_s, y_train)

            p = model.predict_proba(X_test_s)[0][1]
        else:
            # Bayesian fallback
            p = (positives + 0.5) / (len(y_train) + 1.0)

        y_true.append(int(y_test[0]))
        y_pred.append(float(p))

    # Use logloss as performance metric
    try:
        return log_loss(y_true, y_pred, labels=[0, 1])
    except:
        return float("inf")
    
def run_lofo_all_keywords(keyword_datasets):
    all_results = []
    for kw, kw_df in keyword_datasets.items():
        try:
            df_kw = lofo_importance_for_keyword(kw, kw_df)
            all_results.append(df_kw)
        except Exception as e:
            print(f"[WARN] LOFO failed for {kw}: {e}")
    return pd.concat(all_results, ignore_index=True)

ref_lofo = run_lofo_all_keywords(keyword_datasets)
ref_lofo.to_csv("lofo_importance.csv", index=False)