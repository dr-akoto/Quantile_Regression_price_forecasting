# Research project status — evidence and limitations

## Source journey
1. Original UK quantile-regression repository was shared as the published research implementation.
2. Challenge Rebase API endpoint returned 404 HTML; no evidence establishes formal deprecation.
3. Historical CSV was selected for reproducible offline work.
4. Tohya's 2026-11-06 Slack proposal: restore HEFTCom2024 workflow, refresh UK inputs (LCCC/Elexon/NESO), adapt to Japan (JEPX/TEPCO/imbalance), define point-in-time issue conditions, and compare market-relative baselines.

## Delivered now
- Original scripts retained, including their historical leakage risks.
- `code/research_pipeline.py`: market-agnostic 30-minute UTC dataset adapter; validation; strict lagged-price features; expanding-window evaluation with 48-hour label embargo; independent quantile boosting or linear models; monotone quantile rearrangement; weekly-naive benchmark; pinball, interval coverage and width; source hash and reproducibility outputs.
- Unit tests for temporal lags and pinball scoring.

## Not completed / cannot honestly claim
- Published HEFTCom2024 model replication against the full Zenodo competition archive: archive was not supplied and no matching validated point-in-time forecast vintages were available.
- Current UK refresh: no validated LCCC/Elexon/NESO extracts with harmonized product definitions, release times and revisions.
- Japanese operational model: no verified JEPX/TEPCO/imbalance historical extracts or market-specific issue-time contract; UK target columns must **not** be relabeled as Japanese.
- Forecast covariates such as demand/wind/solar **actuals** are excluded intentionally; using historical realized values as if known day-ahead is leakage. A production adapter must ingest forecasts with `issued_at` timestamps and as-of joins.
- Historical 48-hour publication delay is a conservative modeling assumption, not a verified market-specific release schedule.
- Model comparison uses an elementary weekly naive forecast, not a calibrated quantile baseline. Its pinball losses are comparable, but the baseline is not a proper probabilistic benchmark; add seasonal empirical quantiles before paper claims.
- No live API integration, deployment, or production monitoring.

## Next research steps
1. Download and checksum HEFTCom2024 archive; document exact target and forecast vintage schemas.
2. Define issue time per market, timezone, delivery horizon, target unit and publication lag; reject any covariate with later `issued_at`.
3. Reproduce published feature and training protocol without backfill or revised actuals; compare with original reported metrics.
4. Integrate new UK feeds with schema and revision audits; evaluate on newer holdout periods.
5. Acquire Tokyo-area JEPX/TEPCO/imbalance data; normalize timezone, price units, product and settlement period definitions; build within-market baseline.
6. Compare within-market skill scores, reliability and prediction-interval calibration; report limitations and confidence intervals.
