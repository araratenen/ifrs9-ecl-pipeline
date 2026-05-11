# Step 7 — Observation/Performance Windows + Data Quality Fixes

## 1. Observation and performance window choice

**Observation date:** each loan's `issue_d` (origination date).
**Performance window:** the contractual term of the loan (36 or 60 months).
**Label:** `default_flag` — whether the loan transitioned to default ("Charged Off" or "Default") at any point during its term.

This is a **lifetime PD** formulation: the model produces the probability that a loan defaults at any point during its term, conditional on attributes observable at origination.

The alternative — a **snapshot-based 12-month PD** with monthly behavioral features (the typical approach for a live retail bank) — is not feasible on this dataset. Lending Club provides only **terminal** loan status, not monthly payment history; without monthly observations we cannot construct snapshot features as of an arbitrary date or measure a 12-month-forward PD directly. Conversion of the lifetime PD to a 12-month PD for IFRS 9 Stage 1 ECL is performed downstream using vintage hazard rates.

## 2. Data quality fixes applied

| Step | Action | Rows / cells affected |
|---|---|---:|
| 1.1 | Drop "Does not meet the credit policy" rows (pre-2009 underwriting regime) | 1,988 dropped |
| 1.2 | Set `dti = NaN` where `dti == 999` (sentinel for "not computable") | 38 cells |
| 1.3 | Cap `revol_util` at 100 (over-limit revolvers) | 4,687 cells |
| 1.4 | Winsorize `annual_inc` at p99 = $250,000 | 13,448 cells |
| 1.5 | Drop `fico_range_low < 660` (LC issuance minimum is 660) | 2 dropped |

**Rationale:**
- DNMCP rows belong to a pre-2009 credit policy. Mixing them with the modern regime would contaminate the model's view of current underwriting standards. Dropping also corrects an earlier label inversion where DNMCP:Fully Paid was incorrectly mapped to default = 1.
- `dti == 999` is documented as "not computable". Treated as missing rather than as a real DTI of 999.
- `revol_util > 100%` indicates an over-limit revolving account; capping at 100% preserves the high-utilization signal while removing implausible magnitudes.
- `annual_inc` had a max near $11M against a 99th percentile near $250K — clear data-entry artifacts. Winsorizing preserves the row but caps outlier influence.
- The `fico_range_low < 660` band held only 338 rows but a 99.4% default rate against LC's policy of refusing applicants below 660. These are data anomalies whose retention would distort low-FICO modeling signal.

`dti` is left as NaN. No imputation is applied at this stage — downstream WoE binning will handle missingness explicitly as its own bucket.

## 3. Maturity filter

**`as_of` date:** `2019-04-01` — derived as `max(last_pymnt_d) + 1 month`, used as the proxy for the dataset cutoff.

**Rule:** keep loans where `months_observable = (as_of − issue_d) in calendar months ≥ 24`.

**Rationale (survivorship bias):** the labeled population already excludes loans whose status is `Current`, `Late`, or `In Grace Period`. For recent vintages (2017–2018), most slow-performing loans are still `Current` and were excluded in the labeling step, leaving only loans that defaulted quickly. Using these vintages directly would over-represent fast defaulters and inflate apparent default rates. Requiring at least 24 months of observation lets slow defaulters surface; loans not in default by month 24 typically run to maturity without defaulting (consistent with vintage curves seen in EDA — the curve flattens after ~24 months).

**Rows before:** 1,345,348  
**Rows after:** 1,179,687  
**Rows dropped:** 165,661

**Per-vintage breakdown (counts and default rate, before vs. after):**

```
            rows_before  rows_after  rows_dropped  rate_before_%  rate_after_%
issue_year                                                                    
2007                249         249             0          18.07         18.07
2008               1562        1562             0          15.81         15.81
2009               4716        4716             0          12.60         12.60
2010              11536       11536             0          12.89         12.89
2011              21721       21721             0          15.18         15.18
2012              53367       53367             0          16.20         16.20
2013             134804      134804             0          15.60         15.60
2014             223103      223103             0          18.45         18.45
2015             375546      375546             0          20.19         20.19
2016             293105      293105             0          23.29         23.29
2017             169321       59978        109343          23.13         23.13
2018              56318           0         56318          15.76           NaN
```

Pre-filter rates for 2017–2018 are inflated relative to mature vintages because only fast defaulters are present in the labeled set; post-filter, those years are either reduced to a smaller, more representative sub-vintage or removed entirely.

## 4. Feature classification

Three explicit lists, by downstream use.

### PD model inputs (origination features only — 28 features)

```
[
  "loan_amnt",
  "funded_amnt",
  "term_months",
  "int_rate",
  "installment",
  "grade",
  "sub_grade",
  "purpose",
  "annual_inc",
  "dti",
  "fico_range_low",
  "fico_range_high",
  "emp_length_years",
  "home_ownership",
  "verification_status",
  "addr_state",
  "delinq_2yrs",
  "inq_last_6mths",
  "open_acc",
  "pub_rec",
  "revol_bal",
  "revol_util",
  "total_acc",
  "pub_rec_bankruptcies",
  "mort_acc",
  "tax_liens",
  "application_type",
  "earliest_cr_line"
]
```

### Identifiers and dates (3 columns; for joins, not features)

```
[
  "id",
  "issue_d",
  "last_pymnt_d"
]
```

### Outcome / downstream-use only (13 columns; never as PD inputs)

```
[
  "last_fico_range_low",
  "last_fico_range_high",
  "loan_status",
  "total_pymnt",
  "total_rec_prncp",
  "total_rec_int",
  "recoveries",
  "collection_recovery_fee",
  "out_prncp",
  "chargeoff_within_12_mths",
  "collections_12_mths_ex_med",
  "hardship_flag",
  "debt_settlement_flag"
]
```

### Label

`default_flag`

**Why outcome columns cannot be used as PD model inputs (data leakage):** outcome columns encode information observable only **after** the loan has run — total payments received, recoveries collected, last-credit-pull FICO, hardship and settlement flags, etc. A model trained on these features would achieve near-perfect in-sample accuracy by reading off the answer (e.g., a loan with high `recoveries` and `last_fico_range_low ≈ 0` is obviously charged off), but would not generalize at origination time when none of these are known. They are retained in the parquet only because the LGD and EAD models in subsequent steps require them.

## 5. Output

- **Path:** `data/loans_modeling_ready.parquet`
- **Final rows:** 1,179,687
- **Final columns:** 48
- **Default rate:** 19.87%

**Default rate by grade (post-filter):**

```
grade
A     6.00
B    13.24
C    22.22
D    30.21
E    38.59
F    45.26
G    49.85
```

**Default rate by FICO band (post-filter):**

```
fico_range_low
660-700    23.39
700-740    15.84
740-780    10.07
>=780       6.48
```

## 6. Limitations

- The model produces **lifetime PD** because that is what the label measures. Conversion to **12-month PD** for IFRS 9 Stage 1 ECL is performed downstream using vintage hazard rates (transition matrices fit by year-of-origination).
- A live bank would build a snapshot-based 12-month PD with monthly behavioral features (payment trends, utilization changes, late-payment counts). The Lending Club dataset's terminal-status structure does not provide monthly payment history, so this approach is not available here. The lifetime-PD-then-convert workaround is the standard treatment for terminal-status retail credit datasets.
- DNMCP loans were dropped to avoid mixing the pre-2009 (looser) and post-2009 credit-policy regimes in training data. The remaining sample reflects post-2009 underwriting only.
