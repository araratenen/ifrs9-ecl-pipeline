# Power BI Dashboard Specification — IFRS 9 ECL Model (LendingClub Consumer Portfolio)

**Audience.** A Big 4 audit/risk reviewer or interviewer with ~10 minutes of attention. Strong financial instincts; limited time for technical detail.
**Goal.** Communicate three headline ECL numbers, why they differ, and what drives them. Not a data dump.
**Scope.** Four pages, five backing tables, financial-reporting visual style.

---

## Section 1 — Setup

1. **Open Power BI Desktop** (latest version).
2. **Click "Get Data" → "Text/CSV".** Import each of these from `/Users/ostappolukainen/Desktop/ProjRED/data/dashboard/`:
   - `loans_summary.csv`
   - `headline_metrics.csv`
   - `discrimination_metrics.csv`
   - `calibration_table.csv`
   - `sensitivity_table.csv`
3. **In the "Model" view, define relationships:**
   - `loans_summary[id]` is the primary key for the per-loan facts table.
   - The aggregate tables (`headline_metrics`, `discrimination_metrics`, `calibration_table`, `sensitivity_table`) are filtered independently — no direct relationship to `loans_summary` is required for the visuals on Pages 1–4.
   - Optional: create a `vintage_bucket` dimension table if you want a shared vintage slicer across pages.
4. **Set data types:**
   - All ECL columns (`total_ecl`, `ecl_total`, `ecl_final`, `ecl_regulatory`, `ecl_total_*`) → **Currency (USD)**.
   - Percentage columns (`ecl_ratio_pct`, `change_from_baseline_pct`, `stage2_share_pct`, `default_rate`) → **Decimal number**, format as percentage with 2 decimal places.
   - `issue_d` → **Date** (M/d/yyyy).
   - `pd_*`, `lgd_*`, `auc`, `gini`, `ks` → **Decimal number**.
   - `id`, `grade`, `sub_grade`, `purpose`, `home_ownership`, `addr_state`, `vintage_bucket`, `loan_status` → **Text**.
5. **Configure import.** For `loans_summary.csv` (1.18M rows), use Power BI Desktop's default Import mode — it will compress to ~30–50 MB in memory.

---

## Section 2 — Page 1: Executive Overview

### Layout

3 KPI cards across the top, two charts beneath.

### KPI cards (top row, left → right)

| Card | Metric | Subtitle | Formatting |
|---|---|---|---|
| 1 | **Baseline ECL** = `$403.5M` | "Step 12 model output, no forward-looking adjustment" | Card visual; gray background |
| 2 | **Data-Driven Overlay ECL** = `$390.8M` | "IFRS 9 mechanically applied; inverted direction documented" | Card visual; gray background |
| 3 | **Regulatory Overlay ECL** = `$484.5M` | "Recommended for IFRS 9 reporting" | Card visual; **highlighted** in primary color (navy) |

Source: `headline_metrics.csv` filtered to `breakdown = "all"`, `segment = "all"`, three rows for `version ∈ {baseline, data_overlay, regulatory}`.

### Chart 1 (left, lower half): "ECL by Stage — All Three Versions"

- **Visual type:** Stacked horizontal bar chart.
- **Y-axis:** Three bars, one per version (baseline, data_overlay, regulatory).
- **X-axis:** Total ECL.
- **Stack:** Stage 1 / Stage 2 / Stage 3 (color-coded).
- **Tooltip:** Show count and ECL per stage.
- **Source:** `headline_metrics.csv` filtered to `breakdown = "stage"`.

### Chart 2 (right, lower half): "ECL by Grade — All Three Versions"

- **Visual type:** Grouped column chart.
- **X-axis:** Grade A → G.
- **Y-axis:** Total ECL (USD).
- **Group:** Three bars per grade for the three versions.
- **Color:** Match the version palette from the KPI cards.
- **Source:** `headline_metrics.csv` filtered to `breakdown = "grade"`.

### Slicer (top-right corner)

Vintage filter (multi-select): `pre-2014`, `2014`, `2015`, `2016`, `2017+`.
**Source:** `loans_summary[vintage_bucket]`.

---

## Section 3 — Page 2: PD Model Performance

### Layout

4 sections in a 2×2 grid.

### Top-left: "Aggregate Discrimination" (KPI strip)

| KPI | Value |
|---|---:|
| AUC | `0.7059` |
| Gini | `0.4118` |
| KS | `0.2974` |
| Default rate (test) | `23.26%` |
| Test cohort n | `353,083` |

**Source:** `discrimination_metrics.csv` filtered to `dimension = "aggregate"`.

### Top-right: "Out-of-Time Stability" (line chart)

- **X-axis:** Vintage year (2016, 2017).
- **Y-axis:** AUC.
- **Annotation:** "AUC variation < 0.05 → stable" (set as text box).
- **Source:** `discrimination_metrics.csv` filtered to `dimension = "vintage"`.

### Bottom-left: "Reliability Diagram" (clustered column chart)

- **X-axis:** Decile 1–10.
- **Y-axis:** PD value.
- **Two columns per decile:** `predicted_pd` (light) and `observed_pd` (dark).
- **Annotation:** Decile MAD = `0.0276`.
- **Source:** `calibration_table.csv` filtered to `breakdown = "decile"`.

### Bottom-right: "Calibration by Grade" (clustered column chart)

- **X-axis:** Grade A → G.
- **Y-axis:** PD value.
- **Two columns per grade:** predicted vs observed.
- **Source:** `calibration_table.csv` filtered to `breakdown = "grade"`.

### Slicer (top-right)

Filter by `dimension` (vintage / grade / aggregate) for the discrimination KPIs, or by `breakdown` for the calibration tables.

---

## Section 4 — Page 3: Sensitivity & Stress

### Layout

3 panels in a vertical stack.

### Panel 1 (top): "SICR Threshold Sensitivity"

- **Visual type:** Bar chart.
- **X-axis:** SICR rule (`1.25x`, `1.5x`, `2.0x_current`, `2.5x`, `3.0x`, `abs_pd_5pct`, `all_stage1_floor`, `all_lifetime_ceiling`).
- **Y-axis:** Total ECL.
- **Highlight:** the `multiplier_2.00x_current` bar in primary color (navy); others in light gray.
- **Annotation:** "absolute_pd_5pct jumps Stage 2 share to 80%" (call-out).
- **Source:** `sensitivity_table.csv` filtered to `analysis_type = "SICR_threshold"`.

### Panel 2 (middle): "Overlay Weight Sensitivity"

- **Visual type:** Column chart.
- **X-axis:** Weight set (`60_30_10`, `50_30_20_current`, `40_40_20`, `33_33_33`, `40_30_30`, `30_40_30`, `100_baseline`, `100_adverse`, `100_severe`).
- **Y-axis:** Final ECL (data-driven overlay version).
- **Highlight:** `50_30_20_current` in primary color.
- **Source:** `sensitivity_table.csv` filtered to `analysis_type = "overlay_weights"`.

### Panel 3 (bottom): "Single-Feature Stress"

- **Visual type:** Horizontal bar chart.
- **Y-axis:** Feature shock (`sub_grade=G5`, `int_rate=30.99`, etc.).
- **X-axis:** ECL change % from baseline.
- **Color:** Bar shaded by magnitude — red for highest impact, gray for low.
- **Annotation:** Note that `sub_grade` dominates at +9.88% (consistent with IV ranking).
- **Source:** `sensitivity_table.csv` filtered to `analysis_type = "single_feature_stress"`.

### Slicer (top-right)

Filter by `analysis_type` to switch between the three panels' source data when needed.

---

## Section 5 — Page 4: The Macro-Overlay Finding

### Layout

Text panel left (50% width), chart panel right (50% width).

### Text panel (left)

```
THE MACRO-OVERLAY FINDING

Step 8 documented an unusual property of LendingClub data:
when unemployment rises, LC tightens credit standards
within the same year, partially offsetting the macro→default
relationship.

This propagates through the model. The PD model's coefficient
on unrate is -0.41 (with optbinning's WoE convention), meaning
the model has learned "high unemployment at origination →
better-quality borrowers chosen → lower predicted PD".

When we shock unrate +5pp in the severe stress scenario, the
model produces *lower* PDs, leading to a data-driven overlay
ECL that is 3.15% BELOW baseline — the opposite of IFRS 9's
expected direction.

This is mechanically correct on this dataset, but would not
pass production review. The regulatory overlay (right) replaces
the data-derived macro coefficients with conventional regulatory
stress-test sensitivities (+0.18 log-odds per pp unrate, +0.05
per −pp HPI YoY), producing the +20% adjustment IFRS 9 expects.

RECOMMENDATION: Report regulatory overlay $484.5M for external
purposes. Document data-driven overlay $390.8M as an internal
finding for model risk discussion.
```

Format: monospace text box, dark text on light background, 11pt.

### Chart panel (right)

- **Visual type:** Grouped column chart.
- **X-axis:** Scenario (Baseline, Adverse, Severe).
- **Two bars per scenario:**
  - Bar A: "Data-driven ECL" (gray)
  - Bar B: "Regulatory ECL" (primary color — navy)
- **Behavior:**
  - Both bars equal at the **Baseline** scenario ($403.5M).
  - Data-driven decreases through Adverse to Severe.
  - Regulatory increases through Adverse to Severe.
  - The crossing visualizes the inversion in 5 seconds.
- **Annotations:** "Baseline (anchor): $403.5M", "Severe data-driven: $376.2M", "Severe regulatory: $630.4M".
- **Source:** Build a measure table in DAX combining `headline_metrics` rows for `breakdown = "all"` per version, plus the per-scenario rows from `validation_regulatory_overlay.json`. Alternatively, hardcode three rows × two columns directly into a calculated table.

---

## Section 6 — Theme and styling

**Color palette (financial reporting):**
- **Navy** `#1E3A5F` — primary, regulatory headline.
- **Teal** `#3F8E9B` — secondary accent, by-grade visuals.
- **Amber** `#D69E2E` — alerts, the data-driven inversion call-out.
- **Gray** `#6B7280` — supporting elements, subtitles.
- **Light gray** `#F3F4F6` — backgrounds, gridlines.

No pastels.

**Number formatting:**
- USD with thousands separators: `$403,536,501` or `$403.5M` for headlines.
- Percentages with 2 decimal places: `2.37%`, `+20.06%`.
- Use `+` sign on positive deltas (`+9.88%`), no `+` on negatives (`-3.15%`).

**Typography:**
- Body: Segoe UI 11pt.
- Section headers: Segoe UI Semibold 16pt.
- KPI numbers: Segoe UI Bold 24pt.

**Headers and footers (every page):**
- Header: "IFRS 9 ECL Model — LendingClub Consumer Portfolio".
- Footer: "as_of: 2019-04-01 | Author: [your name] | Sources: 1.18M LendingClub loans + FRED macro".

**Page navigation:** Use Power BI's bookmark feature to add Next/Previous buttons in the bottom-right of each page.

---

## Section 7 — Final touches

1. **Bookmark each page** for navigation. Add a "Home" button on Pages 2–4 returning to Page 1.
2. **Tooltips on every chart.** Power BI defaults to truncated tooltips — manually enable full numbers (`Format → Tooltip → On`).
3. **Test the slicers.** Vintage filter on Page 1 should propagate to relevant charts (set up Sync Slicers if you want them shared across pages).
4. **Performance check.** Refresh the model — Power BI should load `loans_summary.csv` in under 30 seconds.
5. **Export the .pbix** as `dashboard/ecl_dashboard.pbix`.
6. **Optionally export to PDF** (`File → Export → Export to PDF`) for static viewing in `dashboard/ecl_dashboard.pdf`.

---

## Source data summary

| File | Rows | Purpose |
|---|---:|---|
| `loans_summary.csv` | 1,179,687 | Per-loan fact table (all visuals can drill into loan-level if needed) |
| `headline_metrics.csv` | ~120 | Aggregate KPIs (Page 1 cards + breakdowns) |
| `discrimination_metrics.csv` | ~10 | AUC/Gini/KS by vintage and grade (Page 2) |
| `calibration_table.csv` | ~25 | Decile reliability + by-grade calibration (Page 2) |
| `sensitivity_table.csv` | ~25 | SICR + weights + feature stress (Page 3) |

---

## Recommended demo flow (10-minute interview)

1. **Open Page 1** — point to the three KPI cards. "Three numbers, all defensible, recommendation is the regulatory overlay at $484.5M."
2. **Click Page 4** — explain the macro-overlay finding. "The data-driven number is below baseline because the model learned LC's underwriting reaction. The regulatory overlay corrects for this with textbook coefficients."
3. **Click Page 2** — show discrimination + calibration. "AUC stable across vintages, calibration MAD under 5%."
4. **Click Page 3** — sensitivity. "Headline is robust to SICR threshold and overlay weights; sub_grade dominates feature importance, consistent with IV ranking."
5. **Return to Page 1** — restate the recommendation. "External reporting: $484.5M. Internal: $390.8M as a model risk discussion item."
