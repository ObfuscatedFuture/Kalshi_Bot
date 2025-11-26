# src/align/align_competitors.py

import pandas as pd
from config.companies import get_report_date

def align_competitor_to_target(comp_name, target_name, df_comp, df_target):
    target_dates = sorted([
        (q, get_report_date(target_name, q))
        for q in df_target["quarter"].unique()
    ], key=lambda x: x[1])

    df_comp["report_date"] = df_comp["quarter"].apply(
        lambda q: get_report_date(comp_name, q)
    )

    aligned_quarters = []
    for _, row in df_comp.iterrows():
        d_comp = row["report_date"]
        match = None
        for q_t, d_t in target_dates:
            if d_comp < d_t:
                match = q_t
                break
        aligned_quarters.append(match)

    df_comp["quarter"] = aligned_quarters
    df_comp = df_comp.dropna(subset=["quarter"])
    df_comp = df_comp.drop(columns=["report_date"])

    return df_comp
