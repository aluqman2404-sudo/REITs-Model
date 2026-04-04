# UK Housing Market Model

Research-grade prototype for regional UK housing analysis. The project combines:

- Stage 2 ingestion of housing and macro inputs
- Stage 3 cleaning and panel assembly
- Stage 4 OLS-based calibration
- Stage 5 stochastic simulation
- Stage 6 regional scoring
- Stage 7 Streamlit delivery for consumer and REIT users

The current app is designed as a reviewable research dashboard, not a trading system and not financial advice.

## Current Stage 4/5 assessment

The canonical rebuild is materially stronger than the earlier Stage 4 and Stage 5 artifacts, but the model is **still not fully signed-off bank-grade**.

Main reasons:

- forecast edge over naive baselines is real but still moderate
- expensive southern regions remain conservative in long-horizon baseline simulations
- the valuation anchor is still a transparent heuristic rather than a full structural model
- the scoring layer remains decision-support research logic rather than a backtested rule engine

See [`STAGE4_STAGE5_ASSESSMENT.md`](./STAGE4_STAGE5_ASSESSMENT.md) for the detailed review.
See [`MODEL_UPGRADE_REPORT.md`](./MODEL_UPGRADE_REPORT.md) for the latest backend upgrade summary.

## What changed in the Stage 7 completion pass

- Added a canonical typed config layer in [`src/core/config.py`](./src/core/config.py)
- Added portable path helpers, structured logging, and dataset validation
- Implemented previously stubbed canonical simulation and scoring helpers
- Added validated dashboard loaders that build app-ready Stage 7 cache files
- Rebuilt the Streamlit app around shared view modules instead of page-local logic
- Added tests for data loading, simulation reproducibility, scoring, and Stage 7 smoke coverage
- Added deployment assets for Streamlit Cloud and Docker
- Documented assumptions, limitations, and the Stage 1-6 audit

## Canonical project structure

```text
housing_model/
├── app/                    # Thin Streamlit page wrappers and assets
├── config/                 # JSON config and schema references
├── data/
│   ├── processed/          # Cleaned panel datasets
│   └── outputs/            # Stage 4-7 outputs
├── src/
│   ├── core/               # Config, paths, logging, validation
│   ├── ingestion/          # Source-specific ingestion modules
│   ├── cleaning/           # Cleaning and feature engineering
│   ├── model/              # Historical Stage 4-6 research scripts
│   ├── scoring/            # Canonical scoring helpers
│   ├── simulation/         # Canonical interactive simulation helper
│   └── ui/                 # Validated loaders, charts, explainers, views
└── tests/                  # Critical-path automated tests
```

## Environment setup

```bash
cd housing_model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the app

Preferred:

```bash
./run_app.sh
```

Equivalent direct command:

```bash
streamlit run src/ui/app.py
```

## Rebuild Stage 7 cache files

The Stage 7 app reads validated Stage 1-6 outputs and writes app-ready cache files to `data/outputs/stage7/`.

```bash
python -m src.ui.build_dashboard_cache
```

Outputs:

- `data/outputs/stage7/dashboard_metadata.json`
- `data/outputs/stage7/region_snapshot.csv`

## Build the enriched Stage 4-ready panel

The richer Stage 2/3 panel is stored in `data/processed/master_dataset_v2.csv`. To export the canonical Stage 4-ready input:

```bash
python -m src.data.export_stage4_ready_panel
```

Outputs:

- `data/processed/ols_ready_dataset_v2.csv`
- `data/outputs/stage3/stage4_ready_panel_metadata.json`

## Rebuild the canonical Stage 4-6 artifacts

To regenerate the preferred model outputs used by the dashboard:

```bash
python -m src.model.canonical_rebuild
```

Outputs:

- `data/outputs/stage4_final/sde_parameters_bankgrade.csv`
- `data/outputs/stage4_final/canonical_stage4_diagnostics.csv`
- `data/outputs/stage5c/simulation_summary_bankgrade.csv`
- `data/outputs/stage6/stage7_handoff_bankgrade.csv`
- `data/outputs/stage6/canonical_rebuild_metadata.json`

## Run the validation pack

To benchmark the rebuilt model against simple baselines and inspect stability:

```bash
python -m src.model.validation_pack
```

Outputs:

- `data/outputs/validation/forecast_benchmark_overall.csv`
- `data/outputs/validation/forecast_benchmark_by_region.csv`
- `data/outputs/validation/coefficient_stability.csv`
- `data/outputs/validation/simulation_plausibility.csv`
- `data/outputs/validation/validation_summary.md`

## Test suite

```bash
pytest tests -q
```

Coverage focus:

- dashboard data integrity
- simulation reproducibility
- score range bounds
- Stage 7 page smoke tests
- stamp duty utility logic

## Data and methodology summary

- The app is anchored to the processed panel and Stage 4/5/6 artifacts already present in `data/outputs/`.
- The dashboard prefers `master_dataset_v2.csv` because it contains the richer Stage 2/3 explanatory set.
- The dashboard prefers the rebuilt `*_bankgrade.csv` Stage 4/5/6 artifacts when they are present and validated.
- Stage 4 provides region-level `kappa`, `sigma`, and equilibrium price estimates.
- Stage 5 provides scenario summaries used for ranking and downside context.
- Stage 6 provides regional consumer and REIT base scores.
- The Stage 7 Scenario Lab does not re-estimate the econometrics. It perturbs calibrated parameters for transparent sensitivity analysis.

## Deployment

### Streamlit Community Cloud

1. Push the repository to GitHub.
2. In Streamlit Cloud, create a new app from the repo.
3. Set the main file to `src/ui/app.py`.
4. Use Python 3.12.
5. If private data access is later needed, configure secrets in the Streamlit UI rather than `.env`.

### Docker

Build:

```bash
docker build -t uk-housing-model .
```

Run:

```bash
docker run --rm -p 8501:8501 uk-housing-model
```

The Docker entrypoint launches:

```bash
streamlit run src/ui/app.py --server.port=8501 --server.address=0.0.0.0
```

## Screenshot generation

See [`docs/SCREENSHOT_GUIDE.md`](./docs/SCREENSHOT_GUIDE.md).

## Important caveats

- Scores are decision-support summaries, not fair-value truth.
- The scoring date is tied to the available Stage 6 handoff, which currently reflects July 2023 starting conditions.
- Regional averages are not substitutes for property-level underwriting.
- Scenario Lab overrides are explicitly exploratory.

## Additional documentation

- [`AUDIT_STAGE1_TO_STAGE6.md`](./AUDIT_STAGE1_TO_STAGE6.md)
- [`ASSUMPTIONS_REGISTER.md`](./ASSUMPTIONS_REGISTER.md)
- [`MODEL_LIMITATIONS.md`](./MODEL_LIMITATIONS.md)
- [`STAGE4_STAGE5_ASSESSMENT.md`](./STAGE4_STAGE5_ASSESSMENT.md)
- [`STAGE7_COMPLETION_REPORT.md`](./STAGE7_COMPLETION_REPORT.md)
