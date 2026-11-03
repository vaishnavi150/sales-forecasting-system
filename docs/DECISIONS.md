# Reasoning: problem, assumptions, decisions, trade-offs and evidence

This document explains **why** the system is built the way it is. For **how** to run it,
see the [README](../README.md). For who did what, see [AI_USAGE.md](../AI_USAGE.md).

## 1. The problem in one paragraph

A sales team has open deals at different stages. The manager must answer *"How much
will we book next quarter?"*. Adding up every open deal over-states the answer, because
many deals are lost and many close later than planned. The system has to:

- estimate each deal's chance of being won **in the next quarter**;
- turn that into a forecast with an honest range;
- prove the method on past quarters;
- point to specific deals that need attention.

## 2. Assumptions

| # | Assumption | Why | If wrong |
|---|---|---|---|
| A1 | Data is synthetic, generated from documented rules | No real CRM data available; the spec allows synthetic data | Real data will be messier; accuracy numbers will differ |
| A2 | One currency (INR) in clean data | Avoids exchange-rate logic. Mixed currency is still detected as an error | Need a conversion step before adding amounts |
| A3 | Forecast horizon = next calendar quarter, made on quarter-end dates | A quarter is the usual sales planning period | Horizon is set in one place (`config.py`, `horizon_end()`) |
| A4 | Deal amount stays fixed over a deal's life | Keeps the generator simple | Amount changes would need their own event history |
| A5 | Forecast covers only deals open on the forecast date | That is what the pipeline can tell us | New deals created and won within the quarter are missed. A known source of under-forecast |
| A6 | Stage default probabilities 10/20/50/75% | Common CRM convention | Only used by one baseline; real rates are learned from history |

## 3. Key decisions and trade-offs

### D1. Rebuild "what was known on date D" from an event log
Every stage or close-date change is a **new row** in `stage_history`; nothing is
overwritten. `build_snapshot(D)` replays events up to D.
- **Why:** this is the only reliable way to avoid leakage. The current `opportunity_stage`
  column already says "Closed Won" for deals that were open on D.
- **Trade-off:** more data and more code than a single "current state" table.
- **Evidence:** a test deletes every record after D and checks that all features are
  identical (`test_features_identical_when_future_data_is_removed`, run for 3 dates).

### D2. Ground truth lives in a separate folder
Outcomes (`opportunity_outcomes.csv`) and anomaly labels sit in `data/ground_truth/`.
`feature_engineering.py` never reads them. `label_snapshot()` refuses to attach
outcomes if the forecast quarter is not finished yet.

### D3. Target = "won during the next quarter", not "eventually won"
- **Why:** we forecast bookings **in a period**, so a deal won one quarter late is a miss.
- **Evidence:** the "historical stage conversion" baseline uses eventual win rates and
  over-forecasts by about 100% (WAPE 103%). That is exactly the slippage this target
  captures. It is kept in the comparison as a lesson.

### D4. Time-based backtest, never random splits
For each snapshot D, models train only on earlier snapshots whose quarter had **ended**
by D (`training_rows()`).
- **Trade-off:** the first quarter has no ML forecast. Only 7 quarters can be compared,
  so differences between methods are uncertain.

### D5. Compare four methods and let the numbers choose
CRM stage probability, historical stage conversion, Logistic Regression and Random
Forest. The **total** forecast uses the lowest backtest WAPE; **deal-level** flags use
the lowest Brier score.

| Starter (7 quarters) | WAPE | Bias | Brier | ROC AUC |
|---|---:|---:|---:|---:|
| CRM stage probability | **13.3%** | +10.8% | 0.156 | 0.775 |
| Logistic Regression | 16.7% | +9.9% | 0.130 | 0.876 |
| Random Forest | 19.1% | +11.2% | **0.126** | **0.881** |
| Historical stage conversion | 103.3% | +103.3% | 0.202 | 0.774 |

- **Conclusion:** ML is clearly better at ranking **individual deals**. It is **not**
  better at the starter **quarterly total**. On full data Random Forest is slightly better
  (3.6% vs 4.0% WAPE). We do not claim ML wins in general.
- **Why ML can rank deals better but forecast totals worse:** totals depend on how well
  probabilities add up (calibration and bias), not on ranking. With 100 reps, a few big
  deals dominate each quarter.

### D6. Weight training rows by deal amount
We forecast **value**, so a miss on a ₹1 crore deal matters more than one on a ₹1 lakh
deal. Measured on the starter backtest: LR WAPE 19.9% → 16.7%, RF 23.7% → 19.1%.
- **Trade-off:** probabilities for small deals become slightly less accurate.

### D7. Scenarios: the wider of two ranges
- **Simulation:** each deal wins or loses at random with its probability; 10,000 runs;
  P10/P90.
- **Historical error:** past actual/forecast ratios applied to today's Base.
- **Downside** = the lower of the two; **Upside** = the higher.
- **Why:** the simulation assumes deals are independent. In the starter backtest its
  P10–P90 range contained the actual result in only 4 of 8 quarters (ideal ≈ 80%), so on
  its own it is over-confident.
- **Honesty:** with 7–8 quarters the historical range is a rough guide. With fewer than
  4 quarters the code marks the result **"not calibrated"** instead of presenting it as
  reliable.

### D8. Risk rules must be checked against history
The first slip-risk rule ("model probability < half the stage rate") flagged 228 of
552 deals. Historically, flagged deals slipped 22.5% of the time vs 17% overall, which
is barely useful. The new rule (early stage, close date 45+ days away, never pushed)
flags 116 deals that slipped **40%** of the time. The check runs on every pipeline run
(`slip_rule_backtest()`).

### D9. Careful language about people
Rep summaries report **facts about their deals** ("4 of 6 open deals flagged"). They say
"pipeline review suggested", never "under-performing". This follows the requirement to
make no unsupported claims about individuals.

### D10. One generator, two scales
`config.SCALES` holds starter (100 reps / 5,000 deals) and full (2,000 / 20,000).
Every module calls `get_config()`. Switching scale needs no code change, and **both
scales are tested**. Running full scale is what exposed the quota bug.

### D11. Things deliberately kept simple
- SQLite instead of a database server: one file, zero setup, still enforces keys and
  checks.
- No hyper-parameter tuning: only 7 evaluation quarters, so tuning would overfit the
  backtest.
- `historical_performance` (rep rating) is generated but not used as a feature. It is a
  subjective rating and could encode bias; win rates from actual deals are used instead.

## 4. Testing approach

| Layer | What is tested | Examples |
|---|---|---|
| Generator | Reproducibility, ID formats, keys, joins, reconciliation | Same seed → identical SHA-256 checksums; bookings = won deals |
| Validation | Every check, both directions | All 16 injected errors found; clean data has 0 issues; no false alarms on the sample |
| Leakage | Features and labels | Removing future data changes nothing; labels refused before quarter end |
| Forecasting | Formula, baselines, time-safe training | `amount × p`; training uses only finished quarters; backtest actual = ground truth |
| Metrics | Hand-checkable numbers | MAE/RMSE/WAPE/bias on 3 values; zero actuals give NaN, not infinity |
| Scenarios | Statistics | Certain deals → fixed total; mean ≈ expected value; ordering Downside ≤ Base ≤ Upside |
| Insights | Rules | Each flag on a hand-made deal |
| Database | Schema rules | Negative amount and unknown rep are rejected by SQLite |
| Dashboard | Loads and filters | Streamlit `AppTest`: no exceptions, metrics present, filters work |

Tests use a small "tiny" dataset (900 deals), so all 80 tests run in about 20 seconds.

## 5. How we know it works

1. **80/80 tests pass** on Python 3.11 and 3.13 (Linux). GitHub Actions re-runs the
   tests and the pipeline on **Windows and Ubuntu** for every push; the result is in
   the repository's Actions tab.
2. **The full pipeline runs end to end** from a fresh clone at both scales.
3. **Reproducible:** generating twice gives byte-identical files.
4. **Validation is proven both ways:** it finds 16/16 injected errors with no false
   alarms, finds 0 issues in clean data, and caught a real generator bug.
5. **Forecasts are judged on outcomes that happened after the forecast date**, quarter
   by quarter, and the numbers are reported even where they are unflattering (ML losing
   on starter totals; simulation coverage 50%).

## 6. Limitations

- Synthetic data: the model learns rules we wrote. Real CRM data will behave differently.
- Only 7–8 evaluation quarters, so method rankings and ranges are uncertain.
- Pipeline-only forecast (new in-quarter deals missed); fixed deal amounts; one currency.
- Simulation ignores correlated shocks (economy, region).
- No probability calibration step; no tuning.

## 7. Next steps if this were real

1. Run the same validation on a real CRM export first. Data quality usually dominates.
2. Add a "new pipeline" component (deals created and won within the quarter).
3. Calibrate probabilities (e.g. isotonic) and re-check totals.
4. Correlated simulation (shared region or market factor).
5. Track forecast accuracy every quarter and re-select the method automatically.
