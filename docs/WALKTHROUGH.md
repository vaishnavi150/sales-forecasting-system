# Walkthrough guide (about 10 minutes)

A script for presenting the project live, followed by questions reviewers are likely to ask.

## Before the call
```powershell
.\.venv\Scripts\Activate.ps1
python run_pipeline.py
streamlit run dashboard/app.py
```
Keep these open: the terminal output, the dashboard, `src/snapshots.py`,
`docs/DECISIONS.md` and the GitHub commit history.

## Demo script

**1. The problem (1 min).**
"Adding up the pipeline over-states bookings. I estimate each deal's chance of closing
**next quarter**, multiply by its amount, and give a Downside / Base / Upside range."

**2. Architecture (1 min).** Show the diagram in the README:
generate → validate → snapshot → features → forecast → scenarios → insights → dashboard.
Each step is its own module in `src/`.

**3. Data and validation (2 min).**
- Data is synthetic, with outcomes driven by documented rules (`docs/data_dictionary.md`).
- In the terminal output, step 2: clean data has 0 errors, and 16/16 injected errors are found.
- Dashboard → *Data quality* tab → detection table.
- Story: running the full scale, validation caught a real bug (zero quota for reps joining
  on a quarter's last day). Show that fix commit.

**4. Leakage (2 min). This is the most important part.** Open `src/snapshots.py`:
- `build_snapshot()` replays only events up to the date.
- `label_snapshot()` refuses outcomes before the quarter ends.
- Mention the test that deletes all future data and gets identical features.

**5. Results (2 min).** Dashboard → *Forecast analysis*:
- forecast vs actual chart, accuracy table;
- "ML ranks deals better (AUC 0.88 vs 0.78) but did not beat the simple baseline on
  starter totals, so the system uses the baseline. On full data Random Forest won slightly."

**6. Scenarios and risks (1 min).** *Overview* and *Risks* tabs:
- *Overview*: the range strip (Downside, Base, Upside, quota). The simulation alone was too
  narrow (50% coverage, shown in *Forecast analysis*), so the range also uses past errors.
- *Risks*: pick one flagged deal and read its reasons; mention the slip-rule check (40% vs 17%).
- Use a sidebar filter (e.g. one region) to show the forecast and range recalculating.

**7. AI use (1 min).** "Claude wrote most of the code. I set the requirements and rules,
and reviewed and validated the result. Here is what I checked." Show `AI_USAGE.md`.

## Likely questions

**How do you know there is no data leakage?**
Features are built only from events dated on or before the snapshot date. Outcomes are
in a separate folder, and labels are refused until the quarter has ended. A test removes
all future rows and confirms the features do not change.

**Why is ML not always used?**
The method is picked by backtest WAPE. On starter data the CRM baseline had a lower
WAPE (13.3% vs 16.7%). Only 7 quarters were compared, so I would not claim either is
better in general.

**Why does the historical-conversion baseline do so badly?**
It uses the chance of *ever* winning, but we forecast winning *this quarter*. Slipped
deals make it over-forecast by about 100%.

**How did you choose Downside and Upside?**
The wider of two ranges: a 10,000-run simulation, and past actual-vs-forecast ratios.
The simulation alone covered the actual in only half of the past quarters.

**What would you change for real data?**
Validate the real export first, add new-in-quarter pipeline, calibrate probabilities, and
track accuracy every quarter.

**What did the AI get wrong?**
It made several mistakes, including a zero-quota bug, a weak slip rule and a wrong
anomaly label. The full list is in `AI_USAGE.md`.

**What would you do with more time?**
See `docs/DECISIONS.md`, section 7.
