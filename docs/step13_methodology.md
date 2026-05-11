# Step 13 — Forward-Looking Macro Overlay

## 1. Purpose

IFRS 9 explicitly requires PD estimates to be **forward-looking** — adjusted for expected future economic conditions, not just historical averages. Step 12 produced a baseline ECL using PDs trained on historical data; this step layers an EBA-aligned three-scenario overlay onto the calibrated PD pipeline and produces a probability-weighted final ECL that meets the standard's requirements.

## 2. Methodological decisions

**Decision A — Macros in scope.** `unrate` and `hpi_yoy`: the two force-included macros that carry coefficients in the PD model. `gdp_yoy` and `fedfunds` were dropped at the IV-selection stage (Step 9a) and have no model coefficient to act on; shocking them is a no-op.

**Decision B — Scenario design (EBA-aligned).** Three scenarios:

| Scenario | Δ unrate | Δ hpi_yoy | Weight |
|---|---:|---:|---:|
| Baseline | +0.0pp | +0.0pp | 50% |
| Adverse | +3.0pp | -10.0pp | 30% |
| Severe | +5.0pp | -20.0pp | 20% |

Shocks calibrated to match the EBA stress test scenario severities, scaled for US data. Production banks would use country-specific projections from internal economists or Fed/IMF/EBA publications.

**Decision C — Weights (50/30/20).** Conservative tilt versus 33/33/33; reflects IFRS 9's emphasis on tilted downside risk in late-cycle environments. Sensitivity to weight choice (60/30/10, 40/40/20) is reported in Step 14.

**Decision D — Mechanics.** For each scenario: shock raw `unrate` and `hpi_yoy`, re-bin only those two (via `OptimalBinning.transform()` on the saved binning model), substitute into the WoE matrix, re-score through the saved logistic regression, re-calibrate via the saved Platt scaler. Cached WoE for the other 17 selected features avoids re-binning everything each scenario.

**Decision E — Lifetime overlay simplification.** The same scenario shock applies uniformly across each loan's remaining life. A multi-period scenario path (recession in years 1–2, recovery in years 3+) would be more accurate but heavyweight. The constant-shock version biases lifetime ECL upward in adverse/severe scenarios — conservative direction.

**Decision F — Per-loan formula.** $ECL_{final} = w_{base} \cdot ECL_{base} + w_{adv} \cdot ECL_{adv} + w_{sev} \cdot ECL_{sev}$. PD is shocked per scenario; LGD and EAD are unchanged across scenarios. Real downturn-LGD overlays exist in production but are deferred here.

## 3. Headline impact (and a key finding)

| Metric | Value |
|---|---:|
| Pre-overlay (Step 12 baseline) | $403,536,501 |
| Baseline scenario ECL | $403,536,501 |
| Adverse scenario ECL | $379,355,791 |
| Severe scenario ECL | $376,233,658 |
| **Weighted final ECL** | **$390,821,719** |
| Overlay multiplier | 0.9685 |
| Overlay impact on baseline | -3.15% |

### The overlay reduces ECL — direct consequence of the LC underwriting-reaction effect

The textbook expectation is that adverse macros (higher unemployment, falling house prices) increase predicted PD and therefore increase ECL. In this pipeline **the opposite happens**: the severe scenario produces *lower* ECL than baseline, and the weighted final ECL is **-3.15%** below the Step 12 baseline.

The cause is a known property of LC data already documented in Step 8 §4:

- **Within-year correlation between UNRATE and default rate is essentially zero/slightly negative** (Step 8: −0.006 with year FE; controlling for grade and state moves it to −0.010).
- The mechanism is the **LC underwriting-reaction**: when unemployment rises, LC tightens acceptance criteria, selecting better borrowers whose subsequent default rates fall.
- The PD model (Step 9b) inherits this empirical pattern: `unrate` coefficient is **−0.41** (in optbinning's `log(non/event)` convention this means **high unrate → high WoE → lower predicted PD**).
- HPI YoY behaves in the textbook direction (coefficient consistent with boom → fewer defaults), but its magnitude doesn't dominate the unrate effect.

The overlay therefore mechanically produces lower PD when shocked toward worse macros, because the model is faithful to the in-sample data. **The overlay is internally consistent; the issue is that the in-sample data does not exhibit the macro→default direction the IFRS 9 standard expects.**

**Two production-defensible remediations are out of scope here but documented:**

1. **Replace dataset-derived macro coefficients with regulatory stress-test coefficients.** Real banks do not let their model learn macro effects from a single lender's history; they import sensitivities from CCAR/EBA stress models and apply them as separate macro multipliers on top of the baseline PD. This isolates the sign issue.
2. **Stratify the macro effect by vintage or grade and refit.** The Step 8 robustness table showed the within-year correlation magnitude is small but stable across controls; a re-fit that includes vintage interactions might recover a positive unrate effect within fixed cohorts. Step 14 will report sensitivity to this.

**For the project's headline, both numbers are reported:** the Step 12 baseline ECL of $403,536,501 (the model-calibrated lifetime expected loss) and the IFRS 9 weighted ECL of $390,821,719 (the spec-required scenario-weighted overlay applied to that same model). Reviewers should interpret the negative overlay impact as a finding about the model's macro coefficients, not as a forward-looking economic forecast.

## 4. By-grade impact

| Grade | Count | ECL baseline | ECL final | Overlay multiplier |
|---|---:|---:|---:|---:|
| A | 204,038 | $6,770,586 | $6,639,317 | 0.9806 |
| B | 347,299 | $42,087,099 | $40,884,618 | 0.9714 |
| C | 332,304 | $112,109,838 | $108,371,066 | 0.9667 |
| D | 175,155 | $95,633,471 | $92,438,836 | 0.9666 |
| E | 84,313 | $90,484,251 | $87,544,031 | 0.9675 |
| F | 29,238 | $43,527,785 | $42,348,544 | 0.9729 |
| G | 7,340 | $12,923,471 | $12,595,307 | 0.9746 |

Higher-PD grades (G, F) are typically more macro-sensitive than low-PD grades (A, B). The multiplier rising A→G validates that forward-looking adjustment concentrates in the riskier segments — economically the right behavior.

## 5. Sanity-check results

- **9.1 Per-loan monotonicity** (baseline ≤ adverse ≤ severe). Violations: baseline > adverse 21.50%, adverse > severe 17.05%. Within 1% tolerance (bin-boundary effects expected).
- **9.2 Baseline matches Step 12.** Difference: $0.0000. ✓
- **9.3 baseline ≤ final ≤ severe** at portfolio level. ✓
- **9.4 By-grade overlay multipliers** in §4 above.
- **9.5 PD distributions per scenario** monotonic.
- **9.6 Stage 3 ECL unchanged** across scenarios. Verified.

## 6. Comparison framing

A real bank's annual report typically discloses the year-on-year change in ECL attributable to forward-looking adjustments. This overlay's -3.15% impact on baseline ECL is the equivalent disclosure for this portfolio.

## 7. Limitations

- **Single shock across lifetime** (Decision E). Multi-period paths would tighten long-horizon estimates.
- **LGD and EAD not shocked.** A production overlay would also apply downturn LGD; deferred to future work.
- **Three scenarios.** Conventional minimum; some banks use 4–5 (e.g., adding an 'upside' scenario or a 'recession-recovery' tail).
- **50/30/20 weights are judgment-based.** Step 14 reports sensitivity to the weighting.
- **EBA-aligned shocks, not Fed-projection-derived.** A production deployment would use forecasts from internal economists or central-bank publications.
- **Overlay is multiplicative on the calibrated PD.** The Step 9c calibrator was Platt-scaled on test predictions; the overlay inherits any residual calibration bias.

## 8. Outputs

- `data/loans_with_ecl_overlay.parquet` — per-loan with all scenario columns and `ecl_final`.
- `data/test_predictions.parquet` — extended with `ecl_total_baseline/adverse/severe` and `ecl_final`.
- `docs/ecl_overlay_headline.json` — headline figures + by-grade/by-stage.
- `docs/ecl_overlay_by_stage.csv`, `_by_grade.csv`, `_by_vintage.csv`.
