# Assumptions Register

## Data treatment

1. The canonical dashboard uses `data/processed/master_dataset_v2.csv` when present, because it contains the richer Stage 2/3 explanatory set required for a credible Stage 4/5 review.
2. Monthly regional averages are treated as the decision surface; property-level heterogeneity is out of scope.
3. CPI is the deflator for house prices, earnings, and rents.
4. Cleaning-stage interpolation and forward-fill rules from the Stage 3 scripts are accepted as the baseline processed panel assumptions.

## Calibration

1. `data/outputs/stage4_final/sde_parameters_bankgrade.csv` is preferred when present, because it is the canonical rebuild on the enriched Stage 2/3 panel.
2. Region-level `kappa`, `sigma`, and `mu_equilibrium` are taken as calibrated research outputs rather than re-estimated inside the app.
3. The app surfaces Stage 4 outputs but does not rerun OLS or IV estimation interactively.
4. The Stage 4 output is treated as a prototype calibration artifact rather than a signed-off bank-grade parameter set.

## Simulation

1. The interactive Scenario Lab uses a log-OU process to preserve positivity and numerical stability.
2. The live simulation uses explicit random seeds for reproducibility.
3. Scenario overrides perturb fair value, volatility, and drift around calibrated values; they do not re-identify structural parameters.
4. The Stage 5 summary remains the canonical source for the named scenario ranking displayed to users.
5. For UI safety, Stage 4 monthly sigma is annualised and blended with realised and Stage 5-implied volatility before being passed into the interactive SDE.
6. When the canonical rebuild has stored sigma directly at annual frequency, the app preserves that unit and does not re-annualise it.

## Scoring

1. Stage 6 handoff scores are treated as the regional base signal.
2. Consumer scoring overlays affordability and downside context onto the Stage 6 base score.
3. REIT scoring overlays yield hurdle and downside context onto the Stage 6 base score.
4. Scores are bounded to `[0, 100]` and translated into `Buy / Hold / Don't Buy` bands using centralized thresholds.
5. No score is presented without contributors, downside context, or a limitations note.

## UI and presentation

1. The app is calibrated to the available Stage 6 handoff date rather than claiming live-now forecasting precision.
2. Scenario Lab output is exploratory and explicitly labeled as such.
3. The dashboard is intended for wide desktop use first, with responsive fallback for smaller screens.
