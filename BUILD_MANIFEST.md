# BUILD MANIFEST
## UK Housing Market Model — v0.2.0

This is the first document a new reviewer should read.
It describes every stage of the canonical pipeline, the exact output
files produced, and how to run the model end-to-end.

---

## Repository layout

```
housing_model/
├── config/
│   └── parameters.json          # Single authoritative configuration file
├── data/
│   ├── raw/                     # Source data files (not committed)
│   ├── processed/               # Cleaned panels produced by Stages 2–3
│   └── outputs/                 # Stage 4–7 artifacts (one sub-directory per stage)
├── src/
│   ├── core/                    # Shared config, paths, validation helpers
│   ├── ingestion/               # Stage 2 — raw data downloaders
│   ├── data/                    # Stage 3 — panel construction
│   ├── model/                   # Stage 4 — calibration; validation runners
│   ├── simulation/              # Stage 5 — SDE Monte Carlo
│   ├── scoring/                 # Stage 6 — signal scoring engine
│   └── ui/                      # Stage 7 — Streamlit dashboard
├── tests/                       # Pytest test suite (46 tests)
├── run_app.sh                   # Convenience launcher for the dashboard
├── setup.sh                     # One-time environment setup
└── BUILD_MANIFEST.md            # This file
```

---

## Canonical pipeline entry point

```
src/model/canonical_rebuild.py
```

A single invocation covers **Stages 3–6** in sequence.
Run it from the `housing_model/` directory:

```bash
# Activate the virtual environment first
source .venv/bin/activate

python -m src.model.canonical_rebuild
```

This writes all Stage 3–6 output files listed below and prints a
JSON summary of the run to stdout.

`build_canonical_panel()` (Stage 3) is called automatically as the
first step inside `build_stage4_parameters()`, so there is no need
to invoke Stage 3 separately.

---

## Canonical app entry point

```
src/ui/app.py
```

The dashboard is a multipage Streamlit application.
Two equivalent ways to start it:

```bash
# Option A — convenience script (sets theme and port)
./run_app.sh

# Option B — direct invocation
streamlit run src/ui/app.py
```

The app loads pre-built Stage 4–7 artifacts from `data/outputs/` on
startup via `src/ui/loaders.py::load_dashboard_data()`.
Stage 7 cache files (`region_snapshot.csv`, `dashboard_metadata.json`)
are built automatically on first load if absent.
All Stage 4–6 outputs must already exist before launching the app.

---

## Stage execution order

Stages must be run in the order shown. Each stage depends on the
outputs of all preceding stages.

| # | Stage | Entry point | How to run |
|---|---|---|---|
| 1 | Environment setup | `setup.sh` | `./setup.sh` (once only) |
| 2 | Data ingestion | `src/ingestion/run_ingestion.py` | `python -m src.ingestion.run_ingestion` |
| 3 | Canonical panel build | `src/data/build_canonical_panel.py` | Auto-invoked by Stage 4 |
| 4 | OLS calibration & fair value | `src/model/canonical_rebuild.py` | `python -m src.model.canonical_rebuild` |
| 5 | SDE Monte Carlo simulation | (same entry point as Stage 4) | (same command) |
| 6 | Scoring handoff | (same entry point as Stage 4) | (same command) |
| 7 | Dashboard cache | `src/ui/build_dashboard_cache.py` | Auto-invoked by app; or `python -m src.ui.build_dashboard_cache` |

Stages 4, 5, and 6 are orchestrated by a single entry point
(`canonical_rebuild.py::run_canonical_rebuild()`), which calls
`build_stage4_parameters()` → `build_stage5_summary()` →
`build_stage6_handoff()` in sequence.

---

## Output files by stage

### Stage 3 — Canonical panel

| File | Location | Description |
|---|---|---|
| `master_dataset_canonical.csv` | `data/processed/` | Canonical monthly research panel; 12 regions × 2005–2024; primary input to all model stages |
| `ols_ready_dataset_v2.csv` | `data/processed/` | Stage 4-ready panel with OLS features pre-computed (lags, growth rates, stress indicators) |
| `descriptive_market_panel.csv` | `data/processed/` | Descriptive panel extending beyond the model cut-off for dashboard context pages |
| `canonical_panel_metadata.json` | `data/outputs/metadata/` | Panel provenance: source file, date ranges, `historical_only` flag, source coverage per series |
| `descriptive_panel_metadata.json` | `data/outputs/metadata/` | Provenance for the descriptive extension panel |
| `stage4_ready_panel_metadata.json` | `data/outputs/stage3/` | Column inventory and row counts for the OLS-ready panel |

### Stage 4 — OLS calibration and fair value

Entry point: `src/model/canonical_rebuild.py::build_stage4_parameters()`

| File | Location | Description |
|---|---|---|
| `sde_parameters_bankgrade.csv` | `data/outputs/stage4_final/` | **Primary Stage 4 output.** Per-region SDE parameters: `kappa` (mean-reversion speed), `sigma` (annual volatility), `mu_equilibrium` (fair-value anchor), `gamma_annual_pp` (drift), directional beta coefficients with guardrail flags, and valuation gap |
| `fair_value_panel_bankgrade.csv` | `data/outputs/stage4_final/` | Full panel with fitted fair values, log fair-value gap, and canonical features; consumed by Stage 5 and both validation runners |
| `canonical_stage4_diagnostics.csv` | `data/outputs/stage4_final/` | Per-region OLS diagnostics: R², adjusted-R², Durbin-Watson, residual std, raw vs final (guardrail-clipped) coefficient values, shrinkage weight |
| `canonical_stage4_pooled_coefficients.csv` | `data/outputs/stage4_final/` | Pooled panel-OLS coefficient table for both the fair-value block (`model_block=fair_value`) and the growth equation block (`model_block=growth`) |
| `rate_channel_candidates.csv` | `data/outputs/stage4_final/` | Sign-stability test for all candidate rate-channel features across full, early, and late subsamples; confirms `mortgage_rate_gap_12m_lag1` as the canonical feature |
| `sigma_mapping_diagnostics.csv` | `data/outputs/stage5c/` | Per-region raw, winsorised, and final annual sigma values with winsorisation bounds; written at Stage 4 close to document the volatility mapping |

### Stage 5 — SDE Monte Carlo simulation

Entry point: `src/model/canonical_rebuild.py::build_stage5_summary()`

Model: log-price Ornstein-Uhlenbeck with a time-varying mean path.
10,000 paths per region-scenario pair over a 60-month horizon.

| File | Location | Description |
|---|---|---|
| `simulation_summary_bankgrade.csv` | `data/outputs/stage5c/` | **Primary Stage 5 output.** 5 scenarios × 12 regions; columns include `median_5yr_growth`, `p10` through `p90`, `prob_terminal_loss_10pct`, `prob_terminal_loss_5pct`, scenario regime probabilities, and the mean-path anchor |
| `baseline_return_model_coefficients.csv` | `data/outputs/stage5c/` | Pooled 5-year log-return model coefficients used to anchor the baseline target price for each region-scenario |
| `region_plausibility_diagnostics.csv` | `data/outputs/stage5c/` | Per-region comparison of simulated baseline median vs historical 5-year return distribution; includes `within_hist_10_90_band` flag and tail-risk gap |
| `simulation_plausibility_revised.csv` | `data/outputs/validation/` | Same plausibility data written to the shared validation directory for use by the independent validation runner |

### Stage 6 — Scoring handoff

Entry point: `src/model/canonical_rebuild.py::build_stage6_handoff()`

| File | Location | Description |
|---|---|---|
| `stage7_handoff_bankgrade.csv` | `data/outputs/stage6/` | **Primary Stage 6 output.** Per-region consumer and REIT scores (0–100), signal component scores (C1–C3), tail-risk metrics, valuation gap (`pct_above_pstar`), scenario-weighted returns, and SDE parameter snapshot for the dashboard |
| `canonical_rebuild_metadata.json` | `data/outputs/stage6/` | Pipeline run provenance: UTC timestamp, paths to Stage 4–6 outputs, region count, scenario count |

### Stage 7 — Dashboard cache

Entry point: `src/ui/build_dashboard_cache.py`
(Also invoked automatically on first `load_dashboard_data()` call.)

| File | Location | Description |
|---|---|---|
| `region_snapshot.csv` | `data/outputs/stage7/` | Merged region table combining Stage 4–6 outputs with latest market observations and signal-context percentiles; primary data source for all dashboard pages |
| `dashboard_metadata.json` | `data/outputs/stage7/` | Data-freshness record: artifact filenames, coverage dates, `historical_only` flag, source-series coverage; read by governance and freshness checks |

---

## Auxiliary validation (optional — not required to run the app)

These runners produce governance evidence and must be run after Stage 6.
The independent validation runner also requires the Stage 7 cache.

### Validation pack

```bash
python -m src.model.validation_pack
```

| File | Location | Description |
|---|---|---|
| `forecast_benchmark_overall.csv` | `data/outputs/validation/` | RMSE and MAE for the canonical model vs zero-growth, 12m-mean, and AR(1) baselines at the 1-month horizon |
| `forecast_benchmark_by_region.csv` | `data/outputs/validation/` | Same metrics broken out by region |
| `forecast_benchmark_rolling.csv` | `data/outputs/validation/` | Benchmark metrics at both 1-month and 3-month horizons |
| `forecast_bias_by_region.csv` | `data/outputs/validation/` | Signed forecast bias by region and horizon |
| `forecast_directional_accuracy.csv` | `data/outputs/validation/` | Directional hit rate (sign agreement) by benchmark, region, and horizon |
| `coefficient_stability.csv` | `data/outputs/validation/` | Full-sample vs early/late subsample coefficient signs and stability classifications; confirms `mortgage_rate_gap_12m_lag1` is sign-stable and `financial_stress_excess_lag3` is correctly classified as episodic |
| `simulation_plausibility.csv` | `data/outputs/validation/` | Copy of `simulation_plausibility_revised.csv` written to the shared validation location |
| `validation_summary.md` | `data/outputs/validation/` | Human-readable summary of forecast edge, coefficient stability, and simulation plausibility |
| `FORECAST_VALIDATION_NOTE.md` | `data/outputs/validation/` | Structured note on forecast benchmark setup, main findings, bias analysis, and limitations |

### Independent validation runner

```bash
python -m src.model.independent_validation_runner
```

This runner consumes frozen artifact CSVs only — it does not refit any models.

| File | Location | Description |
|---|---|---|
| `artifact_lineage_checks.csv` | `data/outputs/validation/` | SHA-256 hashes and row counts for all canonical Stage 3–7 artifacts; confirms pipeline integrity end-to-end |
| `app_artifact_consistency.csv` | `data/outputs/validation/` | Seven checks confirming the Stage 7 cache matches the canonical Stage 4–6 outputs and that the `historical_only` flag is set |
| `simulation_plausibility_replay.csv` | `data/outputs/validation/` | Independent replay of the plausibility check from the frozen Stage 5 CSV |
| `score_bucket_performance.csv` | `data/outputs/validation/` | Forward returns by signal bucket (Supportive / Mixed / Cautious) at 12m, 36m, and 60m horizons |
| `score_calibration_diagnostics.csv` | `data/outputs/validation/` | Spearman correlation and supportive-minus-cautious return spread by signal and horizon |
| `component_signal_strength.csv` | `data/outputs/validation/` | Evidence classification (`supported` / `partial` / `weak`) for each signal component at each validation horizon |
| `historical_vs_simulated_distribution.csv` | `data/outputs/validation/` | Per-region comparison of historical and simulated 5-year return percentiles (P10 through P90) |
| `simulation_moment_comparison.csv` | `data/outputs/validation/` | Regional and pooled-UK moment comparison: mean, median, tail probability, within-band flag |
| `independent_validation_summary.md` | `data/outputs/validation/` | Governance summary: artifact count, consistency checks passed, strongest signal spread, evidence classification |

---

## How to run the test suite

```bash
# From housing_model/ with the virtual environment active
pytest tests/ -v
```

The suite has **61 tests** and completes in under 5 seconds.
All tests read from pre-built artifact files in `data/outputs/` and
`data/processed/`; none refit models or run simulations.

```bash
# Run a single test file
pytest tests/test_canonical_pipeline.py -v

# Run with short tracebacks (CI-friendly)
pytest tests/ --tb=short -q
```

Key test files and what they guard:

| Test file | What it checks |
|---|---|
| `test_canonical_pipeline.py` | Stage 4 has 12 regions with annual sigma in [0.02, 0.12] and `kappa` > 0; Stage 5 scenario ordering is Recovery_Boom ≥ Baseline ≥ Rate_Shock; Stage 6 scores bounded to [0, 100] |
| `test_validation_pack.py` | Model RMSE beats zero-growth and 12m-mean baselines; `mortgage_rate_gap_12m_lag1` is sign-stable across subsamples; stress feature classified as `episodic_not_identified_late` |
| `test_independent_validation.py` | All app consistency checks pass; artifact lineage files exist; valuation signal is monotonic at 36m and 60m; simulation transparency outputs are populated |
| `test_simulation.py` | Monte Carlo produces positive prices, correct (61 × n_paths) shape, and seed-reproducible paths |
| `test_scoring.py` | Consumer and REIT signal dashboards return valid structure; scores bounded to [0, 100]; label is one of Supportive / Mixed / Cautious |
| `test_stage7_smoke.py` | Stage 7 cache files exist and contain the expected columns and all 12 regions |
| `test_governance_and_freshness.py` | `historical_only` flag is `true` in dashboard metadata; data-freshness fields are present |
| `test_ui_governance.py` | Independent validation runner has no import dependency on any UI module |
| `test_stage4_ready_panel.py` | OLS-ready panel has required columns and no all-NA regions |
| `test_dashboard_data.py` | Dashboard data loads without error; region table has expected columns |
| `test_calibration_drift.py` | Detects inter-calibration drift in sigma, gamma, kappa, and R²; asserts guardrail override frequency < 40%. Uses stored baselines in `data/outputs/stage4/`; skips comparison on first run after baseline creation. Matches out-of-cycle triggers in SIGN_OFF_CHECKLIST.md §4.2 and §4.3. |

---

## Configuration

All model parameters live in `config/parameters.json`.
The file is loaded once at import time via
`src/core/config.py::load_config()` and cached with
`functools.lru_cache`. To reset the cache in tests:
`src/core/config.py::reset_config_cache()`.

Key sections:

| Section | Controls |
|---|---|
| `project` | Region list (12 regions), base year (2000), end year (2024), monthly frequency |
| `artifacts` | Canonical filenames for each stage output; must match the paths written by `canonical_rebuild.py` |
| `model` | SDE type (`log_ornstein_uhlenbeck_terminal_target`), `dt` (1/12), `n_simulations` (10,000) |
| `scenarios` | Five presets (Baseline, Soft_Landing, Rate_Shock, Stagflation, Recovery_Boom): rate shifts in bps, stress/transaction overlays, volatility multipliers, drift adjustments |
| `guardrails` | Directional priors for beta coefficients; annual sigma floor and cap; kappa fallback phi |
| `fair_value` | Regressors for the structural fair-value OLS; annual anchor months; kappa estimation bounds |
| `scoring` | Signal thresholds for Supportive / Mixed / Cautious buckets; component weights for consumer and REIT composites |
| `ui` | Default region and scenario; fan-chart percentiles; scenario labels displayed in the dashboard |
| `controls` | Random seed (for simulation reproducibility); max interactive paths; override warning thresholds |

---

## Environment setup (first time only)

```bash
cd housing_model/
chmod +x setup.sh && ./setup.sh
# Edit .env and fill in API keys before running ingestion
source .venv/bin/activate
```

Python 3.12 is the reference interpreter.
The virtual environment is created at `housing_model/.venv/`.

---

## Quarterly refresh

Run this command once per quarter (or whenever new source data is published) to
pull all raw data, rebuild the canonical panel, and re-run Stages 3–6 in a
single operation:

```bash
cd housing_model/
source .venv/bin/activate
python -m src.ingestion.run_ingestion
```

### What it does

1. **Downloads 11 sources** in dependency order (Land Registry HPI → BoE rates →
   ONS earnings/population → MHCLG supply → ONS rental index/levels → computed
   rental yield).
2. **Validates each output** — required columns present, no nulls in key fields,
   at least N distinct years of data.
3. **Guards against coverage regression** — raises `ValueError` before overwriting
   a sidecar if the new data ends earlier than the previous pull.
4. **Updates release-metadata sidecars** (`data/raw/<series>_release_metadata.json`)
   with `observation_period_end`, pull timestamp, source URL, and publication-lag estimate.
5. **Runs the downstream pipeline** (`src/model/canonical_rebuild.py`) to rebuild
   the canonical panel and regenerate Stages 3–6 outputs.
6. **Runs the pytest suite** and reports pass/fail counts.
7. **Prints a JSON summary** to stdout:

```json
{
  "refresh_date": "2026-03-22",
  "new_panel_end": "2025-10-01",
  "staleness_days": 172,
  "effective_data_as_of": "2025-08-01",
  "series_updated": ["Land Registry HPI", "BoE Base Rate", "..."],
  "series_errors": {},
  "tests_passed": 46,
  "tests_failed": 0,
  "pipeline_ran": true,
  "log_file": "data/logs/ingestion_20260322.log"
}
```

### Flags

| Flag | Effect |
|---|---|
| `--strict` | Stop immediately on the first ingestion error |
| `--no-pipeline` | Download sources only; skip Stages 3–6 and tests |

### Log

Each run appends to `data/logs/ingestion_YYYYMMDD.log` (idempotent — safe to
run multiple times on the same day).

### Next scheduled refresh

Full panel refresh with all 11 sources by **2026-06-30**.
Panel is currently at 2026-01-01 (91 days staleness as of 2026-04-02).
4 sources remain failed (ONS Earnings national/regional, ONS Population, MHCLG
Housing Supply — URL changes and schema mismatches tracked in R001).

---

## Build metadata — v0.2.0

| Field | Value |
|---|---|
| Model version | v0.2.0 |
| Build date | 2026-03-31 |
| Panel coverage | 2005-01-01 to 2026-01-01 (12 regions, monthly, 253 dates) |
| Canonical entry point | `src/model/canonical_rebuild.py` |
| Docker build command | `docker build -t uk-housing-model:v0.2.0 .` |
| Docker Compose | `docker compose up` (file: `docker-compose.yml`) |

### Key artifact SHA-256 hashes

Hashes computed with Python `hashlib.sha256` on the binary file content.
Re-run `python -m src.model.independent_validation_runner` to verify
integrity against these values.

| Artifact | Location | SHA-256 |
|---|---|---|
| `sde_parameters_bankgrade.csv` | `data/outputs/stage4_final/` | `a01b38b368d5f481fedb96473119e7682a8102bf95d50118e1a1e148a70d7ae1` |
| `fair_value_panel_bankgrade.csv` | `data/outputs/stage4_final/` | `ef407f7b9855fac0ece2fab62d520412fa4924fa251a0af0596203bb4da94374` |
| `stage7_handoff_bankgrade.csv` | `data/outputs/stage6/` | `5093c35ad0ef22b07fcc03cf67b49495412750bf6f2b99fb692772d43d01fdcd` |

### Test counts

| Suite | Count | Command |
|---|---|---|
| Full suite (excluding rate_channel_stability) | **69 passed**, 1 skipped, 1 deselected | `pytest tests/ -q --tb=short -k "not test_rate_channel_stability"` |
| Rate channel stability (monitored-open, R010) | Deselected by default | `pytest tests/test_validation_pack.py::test_rate_channel_stability -v` |

---

## Change log — v0.2.0

| Session | Changes |
|---|---|
| 14–16 | CI/CD pipeline, Chow structural-break tests, LR HPI refresh infrastructure, Docker hardening, signal backtests (C3/C4/C5), sign-off framework |
| 17A | Sign-offs completed (9 areas APPROVED by Luqman — Student 2026-03-31); C2 (downside_probability) retired from composite; data freshness threshold updated to 210 days |
| 17B | IQR prior implemented for Wales/Scotland/NI/London/South East; sigma raised iteratively to ensure sim 5-yr IQR ≥ 35% of historical |
| 17C | Panel extended to 2026-01-01 (3036 rows, 253 monthly dates); forward-fill governance added (FLOW_SERIES_FFILL_LIMIT=6); scoring_date advanced from 2025-10-01 to 2026-01-01 |
| 17D | OLS replay validation (max_deviation=0.0, passed); subsample OLS validation (train pre-2018, holdout 2018–2026, directional_12m=65.6%); PTI/PTR blend weight estimation (data-driven w=0.30, logged to parameters.json) |
| 17E | Monitoring config (`configure_monitoring()` in logging_utils.py); docker-compose.yml; BUILD_MANIFEST.md updated with artifact hashes and changelog |
