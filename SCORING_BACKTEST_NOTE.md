# Scoring Backtest Note

## Decision layer change

The repository no longer treats Stage 6 buckets as action advice.

Buy/Hold/Don't Buy style labels were downgraded to:
- `Supportive`
- `Mixed`
- `Cautious`

These are descriptive research buckets, not advice labels.

## What was backtested

The independent validation runner backtests a valuation-led historical signal using the canonical fair-value panel and realized forward returns.

Outputs:
- `data/outputs/validation/score_bucket_performance.csv`
- `data/outputs/validation/score_calibration_diagnostics.csv`

## Findings

- 12m evidence is weak and non-monotonic
- 36m evidence is positive and monotonic
- 60m evidence is strongest, with a supportive-minus-cautious spread of about `36.26` percentage points

## Governance interpretation

- valuation-led descriptive bucketing is defensible
- user-specific affordability overlays remain descriptive only
- REIT and consumer composite indicators should not be presented as validated timing rules
