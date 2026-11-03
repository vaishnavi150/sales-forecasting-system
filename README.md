# Sales Forecasting System with Scenario-Based Uncertainty

> **Status: Phase 2 of 12 (project setup).** Sections marked *(planned)* will be
> filled in as each phase is completed. This README only describes features
> that have actually been built.

## 1. Project overview

A Python system that forecasts future sales bookings from opportunity stage,
deal amount, opportunity age, historical conversion rates, expected close dates
and sales-rep activity. It produces **Downside**, **Base** and **Upside**
forecasts, detects data-quality problems and pipeline risks, and compares
forecasts against actual results.

All data is **synthetic** (computer-generated). Results do not represent any
real company.

## 2. Business problem

Sales managers need to answer: *"How much will we book next quarter?"*
Adding up every open deal overstates the answer, because not every deal is won.
This system estimates each deal's chance of winning, multiplies it by the deal
amount, and gives a range of outcomes instead of a single number.

## 3. Architecture

```
Generate data -> Validate -> Snapshots -> Features -> Forecast -> Scenarios & Evaluation -> Insights & Dashboard
```

| Folder / file | Purpose |
|---|---|
| `config.py` | All settings in one place (scale, seed, dates, stages, thresholds) |
| `data/raw/` | Clean generated tables (CSV) |
| `data/processed/` | Feature tables for modelling |
| `data/sample/` | Small dataset with deliberately injected errors |
| `data/ground_truth/` | Actual outcomes and anomaly labels (never used as features) |
| `src/` | Python modules, one per pipeline stage |
| `dashboard/` | Streamlit app |
| `sql/` | Database schema |
| `tests/` | Automated tests (pytest) |
| `reports/` | Generated reports, metrics and charts |
| `docs/` | Data dictionary and design notes |

## 4. Datasets *(planned - Phase 3)*

## 5. Installation (Windows)

Requires **Python 3.11 or newer**.

```powershell
cd sales-forecasting-system
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Check the setup:

```powershell
python config.py
pytest
```

### Choosing the dataset size

The project defaults to the **starter** scale (100 reps, 500 accounts,
5,000 opportunities). To use the **full** scale (2,000 reps, 2,000 accounts,
20,000 opportunities):

```powershell
$env:SFS_SCALE = "full"
```

## 6. How to generate data *(planned - Phase 3)*

## 7. How to train the model *(planned - Phase 7)*

## 8. How to run the dashboard *(planned - Phase 10)*

## 9. How to run tests

```powershell
pytest
```

## 10. Forecasting methodology *(planned - Phases 6-8)*

## 11. Evaluation metrics *(planned - Phase 7)*

## 12. Limitations *(planned)*

## 13. Future improvements *(planned)*
