
# Kalshi_Bot

ML pipeline to predict whether specific keywords will appear on a company's next earnings call, using historical transcripts, competitor signals, and Google Trends features.

## What it does
- Ingests cleaned transcript text and counts keyword mentions (including synonyms).
- Aligns competitor timelines to the target company.
- Builds rolling/weighted features plus seasonality and Trends signals.
- Trains per-keyword logistic models with Bayesian fallback for sparse data.
- Produces next-quarter keyword probabilities and evaluation artifacts.

## Project layout
- `earnings_calls_main.py`: end-to-end pipeline for earnings calls.
- `projects/earnings_calls/config/`: target/competitors, keywords, and parameters.
- `core/ingest/`: transcript loading and keyword counting.
- `core/features/`: feature engineering and wide->long transform.
- `core/models/`: training, prediction, and evaluation.
- `src/trends/`: Google Trends feature builder + cache.
- `projects/fed_pressers/`: tools for Fed press conference cleaning (separate pipeline).

## Setup
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Data prep
Earnings call transcripts are expected as `.cleaned.txt` files next to the PDFs listed in `projects/earnings_calls/config/companies.py`.
Each PDF path must have a matching cleaned file (same name, `.cleaned.txt` suffix).

For Fed pressers, a Powell-only cleaner exists:
```bash
python -m projects.fed_pressers.tools.cleaning
```

## Run the earnings-calls pipeline
```bash
python earnings_calls_main.py
```

## Outputs
- `keyword_feature_wide.csv`: per-quarter feature table.
- `keyword_feature_long.csv`: long-form features for modeling.
- `lofo_importance.csv`: leave-one-feature-out importance.
- Console output for predictions, LOOCV, and forecast CIs.

## Configuration
- Target company and competitors: `projects/earnings_calls/config/companies.py`
- Keywords and synonyms: `projects/earnings_calls/config/keyword.py`
- Model parameters: `projects/earnings_calls/config/parameters.py`

## Notes
- Google Trends data is fetched and cached under `cache/trends`. First run may require network access.
- fed press conference functionality is a WIP and is currently being implemented (planned before Jan 2026 Press Conference)
