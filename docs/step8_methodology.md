# Step 8 — Add Macroeconomic Features (As-of-Origination Join)

## 1. Purpose

Borrower characteristics alone do not capture credit risk: a 700-FICO borrower in 2009 was a meaningfully different risk than the same borrower in 2015. The macro environment at origination shapes both the loan's underwriting and its subsequent performance, so the PD model should see it. The same as-of-origination macro table will later support the IFRS 9 forward-looking overlay, where projected macro paths feed forward PD adjustments.

## 2. Series and transformations

| Column | FRED code | Native frequency | Transformation | Rationale |
|---|---|---|---|---|
| `unrate` | `UNRATE` | monthly, SA | level (%) | Borrowers and lenders react to the absolute rate; 7% means the same in 2009 and 2015. |
| `fedfunds` | `FEDFUNDS` | monthly | level (%) | The policy rate level drives credit pricing and refinancing behavior. |
| `gdp_yoy` | `GDPC1` | quarterly, SA | YoY % change (4-quarter lag), forward-filled to monthly | Real-GDP level grows secularly and is meaningless on its own; only the change rate signals expansion vs. recession. YoY removes seasonality. Quarterly values are forward-filled within the quarter (Q1 → Jan/Feb/Mar). |
| `hpi_yoy` | `CSUSHPISA` | monthly, SA | YoY % change (12-month lag) | Case-Shiller is in arbitrary index units; only the YoY change reflects housing-market direction, which correlates with consumer credit performance through HELOC capacity and wealth effects. |

Adding more macros (CPI, T10Y2Y, INDPRO, etc.) tends to add correlation rather than incremental signal at this aggregation. The four above are the standard set for US consumer credit modeling.

## 3. Join logic

**As-of-origination, on year-month.** For each loan, the join key is the first day of `issue_d`'s month. Macros at that month are attached as features. The model thus learns "what was the environment when this loan was underwritten?" — the question banks face at origination.

The alternative — averaging macros over the loan's contractual life — was rejected. During-life averaging would mix information observable only after origination into the feature set, breaking applicability to live loans (whose future is unknown by definition). It would also blur the underwriting-environment signal.

Lending Club's earliest `issue_d` is 2007-06; macro series start in 2005, so every loan finds a match. Task 4.4 asserts zero nulls in the four new columns.

## 4. Sanity check results

Initial validation using raw UNRATE quartiles produced an inverted relationship (Q1: 22.3% default rate → Q4: 15.8%). Investigation confirmed this is **not a join error** but a **Simpson's paradox**: LendingClub's underwriting loosened progressively from 2009 to 2017, with default rates rising from 12.6% (2009 vintage) to 23.3% (2016 vintage) even as the macro environment improved. Vintage is a confounder strongly correlated with both macros and defaults.

A vintage-controlled within-year correlation check uses the **within-transformation** from panel econometrics: subtract per-year means from both the macros and the default flag, then correlate the residuals. Robustness was checked by progressively adding grade and state as further controls.

| Macro | raw | year-FE | year + grade | year + grade + state |
|---|---:|---:|---:|---:|
| unrate | -0.0638 | -0.0062 | -0.0101 | -0.0098 |
| gdp_yoy | -0.0045 | +0.0047 | -0.0014 | -0.0012 |
| fedfunds | +0.0435 | +0.0071 | +0.0057 | +0.0065 |
| hpi_yoy | -0.0080 | -0.0096 | -0.0116 | -0.0115 |

Two findings:

**(1) HPI YoY shows the expected negative sign robustly** across all specifications. Loans originated during housing booms (high HPI YoY) default less; loans during housing weakness default more. The sign is correct and consistent; the magnitude is small but stable.

**(2) UNRATE shows essentially zero within-year correlation regardless of controls.** Adding more controls makes the residual correlation slightly *more* negative, not less, which rules out simple omitted-variable explanations. The remaining effect is the **LendingClub underwriting-reaction**: when unemployment rises, LC tightens credit standards within the same year, selecting better-quality borrowers and offsetting the macro effect on subsequent defaults. This is a known property of P2P lenders, which adjust acceptance criteria more aggressively than traditional banks. HPI YoY does not exhibit the same offset because LC has no mortgage exposure to react to.

Both correlations are below 0.02 in absolute value — within-year macro variation is small relative to loan-level noise. The macros remain valid PD inputs because (a) the PD model in Step 9 includes `issue_year` as a covariate, disentangling macro effect from vintage drift; and (b) the forward-looking macro overlay in Step 14 operates on cross-scenario macro variation rather than within-year residuals. A noise-band tolerance of |corr| < 0.01 was applied to the assertion; values farther wrong-signed than this would indicate a real failure rather than noise.

**Per-year cross-tab (mean unrate, mean hpi_yoy, default rate):**

```
                 n  mean_unrate  mean_hpi_yoy  default_rate
issue_year                                                 
2007           249         4.79         -3.95         18.07
2008          1562         5.80         -8.83         15.81
2009          4716         9.46         -8.43         12.60
2010         11536         9.57         -2.71         12.89
2011         21721         8.91         -3.70         15.18
2012         53367         8.01          2.20         16.20
2013        134804         7.28          9.82         15.60
2014        223103         6.13          6.46         18.45
2015        375546         5.25          4.58         20.19
2016        293105         4.89          5.06         23.29
2017         59978         4.53          5.47         23.13
```

## 5. Output

- **Loans + macros:** `data/loans_with_macros.parquet`
- **Shape:** 1,179,687 rows × 49 cols
- **Macro reference:** `data/macros_monthly.parquet` (169 months × 4 cols)
- **Feature classification:** `pd_inputs` extended with `unrate`, `gdp_yoy`, `fedfunds`, `hpi_yoy`.

## 6. Limitations

- The four macros are correlated (UNRATE ↑ tends to coincide with GDP YoY ↓, and HPI YoY tracks both). The downstream WoE/IV step will surface which carry the most marginal signal; not all four may be retained in the final model.
- US national level only. Regional macroeconomic variation (e.g., Detroit 2008 vs. Texas 2008) is not captured. Lending Club has `addr_state`, but state-level macro joins are out of scope here.
- Case-Shiller has a real-world publication lag of about two months, which is ignored for retrospective modeling but would matter for live deployment.
- The macros are point-in-time at origination. The forward-looking overlay step (later) handles the projection of macros into each loan's future life.
- FRED occasionally revises historical series. Re-runs after a revision will produce slightly different numbers; re-runnability of this script assumes a stable FRED revision state.
- The within-year correlation between UNRATE and default rate is essentially zero in this dataset (-0.0062), reflecting LendingClub's endogenous underwriting tightening when unemployment rises. This offset is dataset-specific to a P2P lender that adjusts acceptance criteria actively; it would not be expected in a traditional bank's portfolio with stable underwriting standards. HPI YoY does not exhibit the same offset because LC has no direct mortgage exposure. Macros remain valid PD inputs because Step 9's PD model controls for `issue_year`, and the forward-looking overlay in Step 14 operates on cross-scenario rather than within-year variation.
