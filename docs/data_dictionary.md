# Data dictionary

All data is **synthetic**, created by `src/data_generation.py` with seed `20261022`.
Dates are `YYYY-MM-DD`. Money is in **INR** (Indian rupees). 1 crore = 1,00,00,000 INR.

Legend: **Req** = required (cannot be empty), **PK** = primary key (unique ID),
**FK** = foreign key (must exist in the referenced table).

## Entity relationships

| Relationship | Cardinality | Notes |
|---|---|---|
| territories → sales_reps | 1 : N | rep's *current* territory |
| sales_reps ↔ territories | N : M over time | bridge table `rep_territory_history` with start/end dates |
| territories → accounts | 1 : N | |
| accounts → opportunities | 1 : N | |
| sales_reps → opportunities | 1 : N | rep active in the account's territory on the creation date |
| opportunities → stage_history | 1 : N | every change is a new row, history is never overwritten |
| opportunities → activities | 1 : N | optional (a new deal may have no activity yet) |
| sales_reps → quotas | 1 : N | one per quarter from the joining quarter |
| opportunities → historical_bookings | 1 : 0..1 | only won deals |
| opportunities → opportunity_outcomes | 1 : 1 | **ground truth**, stored separately |
| forecast_snapshots | (snapshot, opportunity) | one row per open deal per snapshot date |

**Opportunity lifecycle:** Prospecting → Qualification → Proposal → Negotiation → Closed Won,
or → Closed Lost from any open stage. No events are allowed after a closed stage.

## territories (`data/raw/territories.csv`)
| Field | Type | Req | Allowed values / rule |
|---|---|---|---|
| territory_id | text | PK | `TER-001` … |
| territory_name | text | yes | e.g. "South Territory 2" |
| region | text | yes | North, South, East, West, Central |

## sales_reps
| Field | Type | Req | Allowed values / rule |
|---|---|---|---|
| rep_id | text | PK | `REP-000001` |
| rep_name | text | yes | fictional name |
| region | text | yes | region of current territory |
| territory_id | text | FK | current territory |
| team | text | yes | Inside Sales, Field Sales, Enterprise Sales |
| joining_date | date | yes | 85% joined before 2024, 15% during 2024-2025 |
| historical_performance | number | yes | rating 1.0-5.0 from **before** the data period (not used as a model feature) |

## rep_territory_history (bridge table)
| Field | Type | Req | Rule |
|---|---|---|---|
| assignment_id | text | PK | `RTA-000001` |
| rep_id | text | FK | |
| territory_id | text | FK | |
| start_date | date | yes | |
| end_date | date | no | empty = still current. About 10% of reps move territory (same region) in 2025 |

## accounts
| Field | Type | Req | Allowed values / rule |
|---|---|---|---|
| account_id | text | PK | `ACC-000001` |
| account_name | text | yes, unique | fictional company name |
| industry | text | yes | Healthcare, Education, Manufacturing, Retail, Finance, Technology, Logistics, Government |
| region | text | yes | from territory |
| territory_id | text | FK | |
| account_size | text | yes | Small 40%, Medium 32%, Large 20%, Enterprise 8% |

## stages
| stage_id | stage_name | order | default_probability | is_closed |
|---|---|---|---|---|
| STG-01 | Prospecting | 1 | 0.10 | 0 |
| STG-02 | Qualification | 2 | 0.20 | 0 |
| STG-03 | Proposal | 3 | 0.50 | 0 |
| STG-04 | Negotiation | 4 | 0.75 | 0 |
| STG-05 | Closed Won | 5 | 1.00 | 1 |
| STG-06 | Closed Lost | 6 | 0.00 | 1 |

## opportunities (state on the export date, 2026-09-30)
| Field | Type | Req | Units / rule |
|---|---|---|---|
| opportunity_id | text | PK | `OPP-<creation year>-<counter>`, e.g. `OPP-2026-000001` |
| account_id | text | FK | |
| rep_id | text | FK | |
| product | text | yes | Analytics Suite, CRM Cloud, HR Platform, Security Pack |
| opportunity_created_date | date | yes | 2024-01-01 … 2026-09-30, seasonal, growing over time |
| expected_close_date | date | yes | latest value; ≥ created date |
| opportunity_stage | text | yes | one of the 6 stages |
| deal_amount | number | yes | INR, ≥ 0. Log-normal; median 4 lakh (Small) to 1.2 crore (Enterprise). Fixed for the deal's life |
| currency | text | yes | INR only in clean data |
| current_probability | number | yes | 0-1, = stage default probability |
| is_closed | 0/1 | yes | 1 exactly for Closed Won / Closed Lost |
| last_modified_date | date | yes | date of latest history event |

> The final outcome (**actual_outcome**) is intentionally **not** in this table. It is in
> `data/ground_truth/opportunity_outcomes.csv`, so it cannot leak into model features.

## stage_history (event log)
| Field | Type | Req | Rule |
|---|---|---|---|
| stage_history_id | text | PK | `STG-<opportunity_id>-001` |
| opportunity_id | text | FK | |
| change_date | date | yes | ≥ created date, ≤ 2026-09-30 |
| event_type | text | yes | `stage_change` or `close_date_change` (slippage pushed by the rep) |
| stage | text | yes | stage in effect after the event |
| stage_probability | number | yes | default probability of that stage |
| expected_close_date | date | yes | expected close date in effect after the event |

## activities
| Field | Type | Req | Rule |
|---|---|---|---|
| activity_id | text | PK | `ACT-00000001` |
| opportunity_id | text | FK | |
| rep_id | text | FK | owner of the opportunity |
| activity_date | date | yes | between created date and close date |
| activity_type | text | yes | Call, Email, Meeting, Follow-up |
| activity_count | integer | yes | ≥ 1 (activities of the same type on the same day are grouped) |

## quotas
| Field | Type | Req | Rule |
|---|---|---|---|
| quota_id | text | PK | `QUOTA-<rep_id>-2026Q1` |
| rep_id | text | FK | |
| quota_quarter | text | yes | 2024Q1 … 2026Q4 |
| quarter_start, quarter_end | date | yes | |
| quota_amount | number | yes | INR, > 0. Team base × 6% yearly growth × random 0.9-1.1, pro-rated for joiners |
| currency | text | yes | INR |

## historical_bookings (derived)
One row per won opportunity. Booking date = date of the "Closed Won" event.
Amount = opportunity deal amount. Validation checks this reconciliation.

| Field | Type | Rule |
|---|---|---|
| booking_id | text | PK `BKG-000001`, ordered by booking date |
| opportunity_id | text | FK, unique |
| rep_id, account_id, territory_id | text | FK |
| booking_date | date | |
| booking_quarter | text | e.g. 2025Q3 |
| booking_amount | number | INR |
| currency | text | INR |

## opportunity_outcomes (ground truth — `data/ground_truth/`)
| Field | Rule |
|---|---|
| opportunity_id | PK |
| final_status | Won / Lost / Open (Open = not closed by 2026-09-30) |
| close_date | date of Closed Won/Lost event, empty if open |
| won_amount | deal amount if won, else 0 |
| initial_expected_close_date | expected close date when the deal was created |
| slipped | 1 if closed later than first expected, 0 if not, empty if open |

## anomaly_labels (ground truth for the error sample)
| Field | Rule |
|---|---|
| label_id | `ANOM-001` |
| table_name | table containing the broken record |
| record_id | ID of the broken record (snapshots use `snapshot_id|opportunity_id`) |
| issue_type | the validation issue type that should be detected |
| description | what was broken |

## forecast_snapshots (`data/processed/`)
Built by `src/snapshots.py` from stage history up to each snapshot date.

| Field | Rule |
|---|---|
| snapshot_id | `FCST-2026-03-31-001` |
| snapshot_date | quarter end, 2024-09-30 … 2026-09-30 |
| opportunity_id, account_id, rep_id, product, opportunity_created_date | static deal fields |
| stage, stage_probability, expected_close_date | from the latest event on or before the snapshot date |
| stage_as_of_date | date of that event; must be ≤ snapshot_date (leakage check) |
| deal_amount, currency | |
| weighted_amount | deal_amount × stage_probability (reconciliation check) |

## How outcomes are generated (documented rules)
- Hidden win score = −0.55 + 1.2 × rep skill + 0.8 × deal engagement + account-size effect + industry effect + product effect. Win probability = sigmoid(score). Hidden values are never saved.
- Lost deals stop at Prospecting 40%, Qualification 30%, Proposal 18%, Negotiation 12% (shifted later for engaged deals).
- Slip probability = sigmoid(−0.5 − 0.8 × engagement). Slipped deals take 1.15-2.2× the planned cycle.
- When a deal slips, the rep pushes the expected close date with probability equal to their "CRM hygiene" (≈75%); otherwise the deal becomes overdue.
- 30% of won deals are pulled to the last two weeks of their quarter (quarter-end effect).
- Activity rate ≈ 1.5 per week × rep activity level × e^(0.4 × engagement). 75% of lost deals "go dark" 15-60 days before being lost.
- Everything after 2026-09-30 is cut off, like a real CRM export.
