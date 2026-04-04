# Model Limitations

## Scope limitations

1. The model works on regional averages, not individual properties.
2. It does not observe borrower credit quality, mortgage product mix, tax position, or refurbishment needs.
3. The current decision surface is anchored to the historical sample and the available Stage 6 scoring handoff.

## Data limitations

1. Several economic series are interpolated from annual or quarterly frequency to monthly.
2. Housing supply and earnings inputs remain dependent on official series with publication lags and revisions.
3. Raw-source ingestion is not yet governed by a full formal schema layer at every source boundary.

## Econometric limitations

1. Stage 4 contains multiple research scripts and variants; Stage 7 relies on the final output files rather than re-adjudicating every historical specification.
2. Parameter stability can still change if the sample, controls, or structural breaks are revised.
3. Even with diagnostics, omitted-variable bias and regime instability are real risks in housing data.
4. The canonical validation pack shows that the new rate repricing feature is sign-stable, but its magnitude is still sample-sensitive across early versus late subsamples.

## Simulation limitations

1. The interactive Scenario Lab is a calibrated perturbation tool, not a full re-run of the original Stage 5 research workflow.
2. The live simulation is intentionally simplified for responsiveness and transparency.
3. Scenario outputs are conditional on assumed macro paths and should not be interpreted as unconditional forecasts.
4. The rebuilt Stage 5 baseline is materially better than before, but it remains conservative relative to the historical five-year outcome distribution in several high-price regions.

## Scoring limitations

1. Scores compress multiple signals into a single number and therefore lose information.
2. The consumer and REIT scores are structured decision aids, not recommendations.
3. Threshold-based labels can create false precision if read without the component decomposition and scenario notes.

## Engineering limitations

1. The historical Stage 2-6 scripts are still more script-oriented than a full production pipeline.
2. The canonical dashboard path is well-validated, but the entire repo is not yet refactored into one uniform orchestration framework.
3. Some optional dependencies from earlier experiments remain listed for compatibility even though the final Stage 7 path does not require all of them.
