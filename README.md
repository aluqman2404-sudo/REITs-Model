# UK Regional Housing Market Model — v0.2.0

A research-grade stochastic simulation platform for UK residential property across 12 regions. Built for systematic fair-value research and long-horizon scenario analysis by institutional investors and REIT analysts.

> **Disclaimer:** This model is a historical research prototype. It does not constitute regulated financial advice, investment recommendations, or a stress-testing model as defined under PRA SS3/18.

---

## Documentation

| Document | Description |
|---|---|
| [Technical Report (PDF)](./docs/UK_Housing_Model_Report.pdf) | Full mathematical derivations, calibrated parameters, validation results, governance framework |
| [Technical Report (LaTeX source)](./docs/UK_Housing_Model_Report.tex) | Source for the PDF above |
| [Model Validation Summary](./docs/MODEL_VALIDATION_SUMMARY.md) | Sign-off results, backtesting, sensitivity analysis, CI coverage |
| [UI Section Guide](./docs/UI_SECTION_GUIDE.pdf) | Page-by-page dashboard documentation |
| [Sign-Off Checklist](./SIGN_OFF_CHECKLIST.md) | 9/9 governance areas — all APPROVED (2026-04-02) |
| [Model Risk Register](./MODEL_RISK_REGISTER.csv) | 26-item risk register with statuses |
| [Model Limitations](./MODEL_LIMITATIONS.md) | 8 documented material limitations |
| [Build Manifest](./BUILD_MANIFEST.md) | Authoritative build path and canonical pipeline |
| [Assumptions Register](./ASSUMPTIONS_REGISTER.md) | Full list of modelling assumptions |
| [Scoring Backtest Note](./SCORING_BACKTEST_NOTE.md) | Label governance and scoring calibration |
| [Signal Definitions](./SIGNAL_COMPONENT_DEFINITIONS.md) | C1/C2/Yield/Tail component definitions |
| [Interpolated Feature Policy](./INTERPOLATED_FEATURE_POLICY.md) | Earnings staleness and interpolation policy |

---

## Model at a Glance

- **Regions:** 12 (all English regions + Wales, Scotland, Northern Ireland)
- **Data sources:** Land Registry HPI, Bank of England, ONS, HMRC
- **Panel:** 2005–2025, monthly, 2,532 rows × 41 columns
- **Simulation:** Log Ornstein-Uhlenbeck, 10,000 paths × 5 scenarios × 12 regions = 600,000 paths
- **Horizon:** 5 years (60 monthly Euler-Maruyama steps)
- **Validation status:** APPROVED — 9/9 sign-off areas complete

### Key validation results

| Test | Result | Status |
|---|---|---|
| OLS replay deviation | 0.000 | PASS |
| Subsample directional accuracy (12m) | 64.7% | PASS (threshold 55%) |
| Model horse race — Log-OU vs GBM RMSE (12m) | 0.044 vs 0.049 | Log-OU wins |
| CI coverage (pooled, holdout 2018–2026) | 78.7% | MARGINAL |
| Information ratio (36m) | 63.1% | PASS (threshold 55%) |

---

## Seven-Stage Pipeline

```
Stage 1  Ingestion      Land Registry, BoE, ONS, HMRC raw data fetch and cache
Stage 2  Cleaning       Harmonise, interpolate, construct monthly panel
Stage 3  Feature eng.   Multicollinearity diagnostics, lagged macro features
Stage 4  OLS calib.     Fair-value layer + OU parameters (κ, σ, γ) per region
Stage 5  Simulation     10,000-path Euler-Maruyama per region per scenario
Stage 6  Scoring        Consumer and REIT composite scores
Stage 7  Dashboard      Streamlit research UI — 8 pages, 12 regions
```

---

## Project Structure

```
housing_model/
├── docs/                   # Technical report, validation summary, UI guide
├── config/                 # parameters.json, data_schema.json, settings
├── src/
│   ├── core/               # Config, paths, logging, validation
│   ├── ingestion/          # Land Registry, BoE, ONS, HMRC ingestion
│   ├── cleaning/           # Panel construction and feature engineering
│   ├── model/              # OLS, OU estimation, validation, canonical rebuild
│   ├── scoring/            # Scoring engine and backtest
│   ├── simulation/         # SDE and Euler-Maruyama
│   └── ui/                 # Streamlit app, charts, loaders, views
├── tests/                  # 50 tests — governance, freshness, scoring, smoke
├── SIGN_OFF_CHECKLIST.md
├── MODEL_RISK_REGISTER.csv
├── MODEL_LIMITATIONS.md
└── requirements.txt
```

---

## Setup

```bash
cd housing_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run src/ui/app.py
```

## Rebuild canonical model outputs

```bash
python -m src.model.canonical_rebuild
```

## Run the validation pack

```bash
python -m src.model.validation_pack
```

## Run tests

```bash
pytest tests -q
```

---

## Deployment

### Streamlit Community Cloud (free)

1. Push this repo to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** → select `REITs-Model` → branch `main` → main file `src/ui/app.py`.
4. Click **Deploy**.

### Docker

```bash
docker build -t uk-housing-model .
docker run --rm -p 8501:8501 uk-housing-model
```

---

## Important caveats

- Scores are descriptive research labels (Supportive / Mixed / Cautious), not investment advice.
- The model operates on regional averages — not applicable to individual properties.
- Earnings data is interpolated annually; 13-month staleness as of April 2026.
- The Scenario Lab applies coefficient-based perturbations, not a full econometric re-run.
- See [Model Limitations](./MODEL_LIMITATIONS.md) for the full list.
