# How I used AI on this project

**Tool:** Claude (Anthropic), in a Claude Project, using its built-in sandbox to run
code. All prompts are in [`PROMPT_HISTORY.md`](PROMPT_HISTORY.md).

## Summary

| Area | Who did it |
|---|---|
| Problem definition, requirements, tech stack, 12-phase plan, rules for the AI | Me (see `prompts/00_project_instructions.md`) |
| Assignment specification | Given to me (see `prompts/01_requirements_spec.txt`) |
| Architecture proposal, data model, scale design | AI proposed; I accepted |
| Python code (~3,500 lines), SQL schema, tests (~720 lines, 80 tests) | AI wrote all of it |
| Running code and tests, finding and fixing bugs during the build | AI, in its sandbox |
| README, data dictionary, decision and AI-usage documents | AI drafted |
| Review, testing on my own laptop, edits, publishing to GitHub | Me (see the section below) |

**In plain words:** the AI wrote almost all of the code and documentation. My role was
to define the problem and the rules, choose the direction at each step, and then review,
run, question and validate the result.

## Where AI accelerated the work

- **Turning a long specification into a design.** It reconciled my instructions with the
  assignment spec (they disagreed on volumes and entities) and proposed one generator
  with two scales instead of two separate datasets.
- **Writing a realistic data generator.** This covered slippage, stale deals,
  seasonality, territory changes, quarter-end closing, and outcomes driven by
  documented rules instead of random coin flips.
- **Boilerplate at scale:** 30+ validation checks, 80 tests, the SQLite schema, the
  Streamlit dashboard and the documentation.
- **Running experiments quickly.** Examples: comparing four forecasting methods over
  7 quarters, and testing whether weighting by deal amount helps.

## Where I relied heavily on AI

- **Statistical and modelling design.** This covers the target definition, smoothing of
  win rates, the backtest design and how the scenario ranges are built. I understood and
  accepted these choices, but I did not design them myself.
- **Library details** (scikit-learn pipelines, pandas date handling, Streamlit testing).
- **Debugging during the build:** the AI found and fixed most of its own bugs before I
  saw the code (listed below).

## Decisions the AI made that I must be able to defend

These were AI suggestions, not requirements from me. Each is explained in
[`docs/DECISIONS.md`](docs/DECISIONS.md).

1. INR as the only currency in clean data, and a quarterly forecast horizon. These were
   defaults, because I did not answer the Phase 1 questions.
2. The target is "won **during the next quarter**", not "eventually won". This makes
   slippage part of the target.
3. Training rows are weighted by deal amount, because we forecast value. This was
   measured: Logistic Regression WAPE went from 19.9% to 16.7%, Random Forest from
   23.7% to 19.1%.
4. The method is picked automatically by the lowest backtest WAPE. On starter data this
   picks the simple CRM baseline, not ML.
5. Downside/Upside use the wider of two ranges (simulation and historical error),
   because the simulation alone proved too narrow.
6. The slip-risk rule was replaced after it was checked against history (see below).

## Mistakes the AI made, and how they were caught

All of these happened during the build and are visible in the commit history or the
transcript.

| Problem | How it was caught | Fix |
|---|---|---|
| `pandas.Index` has no `.sum()`, so the generator crashed | First run | Converted to NumPy arrays |
| `pd.Timedelta(days=...)` raised deprecation warnings | Running with warnings as errors | Used `pd.Timedelta(n, unit="D")` |
| The label for the injected "duplicate snapshot row" was built **after** duplicating the row, giving a wrong key, so detection showed 15/16 | Comparing validation results with the anomaly answer key | Built the key before duplicating |
| First slip-risk rule flagged 228 of 552 deals. In history, flagged deals slipped 22.5% vs 17% for all deals: barely useful | Back-testing the rule against real outcomes | New rule: flagged deals slipped 40% vs 17% |
| Reps with 3 open deals dominated the "needs review" list | Reading the output | Minimum raised to 4 open deals |
| A test compared floats with `==` | Failing test | `pytest.approx` |
| `use_container_width` is deprecated in Streamlit | Warning in the dashboard test | `width="stretch"` |
| Streamlit tried to send usage statistics to the internet | Network log in the sandbox | Disabled in `.streamlit/config.toml` |
| **A rep joining on the last day of a quarter got a quota of 0** | **Our own validation flagged 2 errors when the full scale was run** | Counted days inclusively; added a regression test |

The last one is the best evidence that the validation layer works: it caught a real bug
in the generator, not just the errors we injected on purpose.

## Dashboard redesign (prompt 6)

The first dashboard was functional but plain. On request, the AI redesigned it: a
forecast range strip as the main visual, interactive charts with tooltips, white
panels, responsive number cards, and a separate Risks tab. It reviewed its own work by
taking browser screenshots at two screen widths and fixed what it saw:

- Streamlit's own styles overrode the big forecast number, so it used more specific CSS.
- The range-strip labels were too small to read, so it resized the drawing.
- Tables were cut off on the right, so it stacked them or removed duplicate columns.
- Win chances showed odd decimals ("0.16%"), so they now show whole percentages.
- The four number cards were squeezed on smaller screens, so they now wrap into two rows.

The dashboard tests were updated for the new layout (3 tests, 80 in total).

## Limits of what the AI could check

- The AI's sandbox runs Linux. It tested on Python 3.11 and 3.13 there, but not on
  Windows. The GitHub Actions workflow runs on Windows and Ubuntu to cover this.
- The AI could not log in to GitHub. I created and pushed the repository myself.
- The commit history was created on 2026-11-03, after the build. It replays the actual
  order of development from the 2026-11-02 session (phase by phase, including the fixes
  above). The commit dates are the dates the commits were made, not backdated. Every
  commit carries a `Co-Authored-By: Claude` line.

## What I personally reviewed, changed and validated

> **TODO (Vaishnavi):** write this section yourself, in your own words. Only list
> things you actually did. Delete any line that is not true. A checklist to work through:

- [ ] Installed and ran `python run_pipeline.py` on my Windows laptop. Result: …
- [ ] Ran `pytest` on my laptop. Result: … passed
- [ ] Ran `python -m src.data_generation --check-reproducibility`. Result: …
- [ ] Opened the dashboard and tried the filters, search and downloads. What I noticed: …
- [ ] Read `src/snapshots.py` and `src/feature_engineering.py`, and can explain how leakage is prevented
- [ ] Checked one forecast by hand (picked an opportunity, computed amount × probability). Opportunity ID: …
- [ ] Checked one injected error in `data/sample/` against `data/ground_truth/anomaly_labels.csv`
- [ ] Questioned or changed something the AI did. What and why: …
- [ ] Parts I do not fully understand yet: …
