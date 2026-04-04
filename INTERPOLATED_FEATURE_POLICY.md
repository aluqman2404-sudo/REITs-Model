# Interpolated Feature Policy

## Problem fixed

The previous pipeline linearly interpolated annual earnings, population, and supply to monthly frequency. That created fake monthly precision and, for annual series, could leak future anchor values into earlier months.

## Canonical policy

1. Canonical Stage 4 no longer relies on interpolated monthly earnings changes.
2. Annual earnings are now handled as an as-of series:
   - last observed annual value only
   - no future annual anchor may influence an earlier month
3. The canonical panel records:
   - `earnings_obs_date`
   - `earnings_staleness_months`
   - `earnings_anchor_flag`
   - `earnings_interpolated_flag`
4. Stage 4 uses `earnings_growth_12m_lag1`, not a fake monthly earnings diff.

## Additional flags

The canonical panel also emits anchor flags for:
- annual affordability observations
- population anchors
- supply anchors

## Scope

This policy materially fixes the canonical Stage 4-7 path. It does not rewrite every older processed file in the repository.
