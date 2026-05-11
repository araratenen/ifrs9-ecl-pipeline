## IFRS 9 ECL Modeling Pipeline — DAG orchestration
##
## Each `step*` target's primary artifact is its sentinel; downstream steps
## depend on those artifacts by path, so `make all` rebuilds only what is
## stale. Phony targets (`pd-pipeline`, `ecl-pipeline`, `qa`, `clean`) group
## logical phases.
##
## Usage:
##   make help            # list targets
##   make all             # full pipeline (Step 7 → final dossier)
##   make pd-pipeline     # through Step 9c (calibrated PD)
##   make ecl-pipeline    # through Step 12 (baseline ECL)
##   make overlay         # through Step 14 (regulatory overlay)
##   make dashboard       # Step 15 dashboard data
##   make docs            # project summary + final dossier
##   make qa              # validator + audit + pytest
##   make test            # pytest unit tests only
##   make clean           # remove ALL generated artifacts (asks first)
##   make clean-docs      # remove only generated docs

PYTHON      ?= .venv/bin/python
PYTEST      ?= .venv/bin/pytest
ROOT        := $(shell pwd)
SRC         := src
DATA        := data
DOCS        := docs
MODELS      := models
DASHBOARD   := $(DATA)/dashboard

# ---------- Inputs ----------

ACCEPTED_LABELED := $(DATA)/accepted_labeled.parquet

# ---------- Step outputs (primary sentinels) ----------

LOANS_READY   := $(DATA)/loans_modeling_ready.parquet
LOANS_MACROS  := $(DATA)/loans_with_macros.parquet
MACROS_MONTH  := $(DATA)/macros_monthly.parquet
TRAIN_PARQ    := $(DATA)/train.parquet
TEST_PARQ     := $(DATA)/test.parquet
TRAIN_WOE     := $(DATA)/train_woe.parquet
TEST_WOE      := $(DATA)/test_woe.parquet
BINNING_PKL   := $(MODELS)/binning_process.pkl
BINNING_JSON  := $(DOCS)/binning_summary.json
PD_LR         := $(MODELS)/pd_logistic.pkl
PD_GBM        := $(MODELS)/pd_xgboost.pkl
TEST_PRED     := $(DATA)/test_predictions.parquet
EVAL_JSON     := $(DOCS)/model_evaluation.json
CAL_LR        := $(MODELS)/pd_logistic_calibrator.pkl
CAL_POST      := $(DOCS)/calibration_post.json
LOANS_LGD     := $(DATA)/loans_with_lgd.parquet
LOANS_EAD     := $(DATA)/loans_with_ead.parquet
LOANS_ECL     := $(DATA)/loans_with_ecl.parquet
ECL_HEADLINE  := $(DOCS)/ecl_headline.json
LOANS_OVERLAY := $(DATA)/loans_with_ecl_overlay.parquet
OVERLAY_JSON  := $(DOCS)/ecl_overlay_headline.json
REG_OVERLAY   := $(DOCS)/validation_regulatory_overlay.json
DASH_SUMMARY  := $(DASHBOARD)/loans_summary.csv
SUMMARY_MD    := $(DOCS)/project_summary.md
DOSSIER_MD    := $(DOCS)/final_project_dossier.md

FC_JSON       := $(DOCS)/feature_classification.json

# ---------- Phony targets ----------

.PHONY: help all data pd-pipeline ecl-pipeline overlay dashboard docs \
        validate audit test qa clean clean-docs init dag

help:
	@awk 'BEGIN{FS=":.*##"} /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[1m%-16s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# default
all: $(DOSSIER_MD)  ## Full pipeline through final dossier

data: $(LOANS_MACROS)             ## Through Step 8 (loans + macros)
pd-pipeline: $(CAL_POST)          ## Through Step 9c (calibrated PD)
ecl-pipeline: $(ECL_HEADLINE)     ## Through Step 12 (baseline ECL)
overlay: $(REG_OVERLAY)           ## Through Step 14 (regulatory overlay)
dashboard: $(DASH_SUMMARY)        ## Step 15 dashboard data
docs: $(DOSSIER_MD)               ## Project summary + final dossier

# ---------- Step rules ----------

# NOTE — make 3.81 lacks grouped-target syntax (`&:`), so multi-output steps
# declare ONE primary target with the recipe; secondary outputs depend on the
# primary with an empty recipe. This makes the recipe run exactly once when
# any output is requested.

# Step 7: clean & feature-classify (primary: LOANS_READY)
$(LOANS_READY): $(ACCEPTED_LABELED) $(SRC)/step7_observation_window.py
	$(PYTHON) $(SRC)/step7_observation_window.py
$(FC_JSON): $(LOANS_READY) ;

# Step 8: macros (primary: LOANS_MACROS)
$(LOANS_MACROS): $(LOANS_READY) $(FC_JSON) $(SRC)/step8_macro_features.py
	$(PYTHON) $(SRC)/step8_macro_features.py
$(MACROS_MONTH): $(LOANS_MACROS) ;

# Step 9a: WoE binning + train/test split (primary: BINNING_JSON)
$(BINNING_JSON): $(LOANS_MACROS) $(FC_JSON) $(SRC)/step9a_woe_binning.py
	$(PYTHON) $(SRC)/step9a_woe_binning.py
$(TRAIN_WOE) $(TEST_WOE) $(TRAIN_PARQ) $(TEST_PARQ) $(BINNING_PKL): $(BINNING_JSON) ;

# Step 9b: PD models (primary: EVAL_JSON)
$(EVAL_JSON): $(TRAIN_WOE) $(TEST_WOE) $(BINNING_JSON) $(SRC)/step9b_pd_model.py
	$(PYTHON) $(SRC)/step9b_pd_model.py
$(PD_LR) $(PD_GBM) $(TEST_PRED): $(EVAL_JSON) ;

# Step 9c: Platt calibration (primary: CAL_POST; mutates TEST_PRED in place)
$(CAL_POST): $(PD_LR) $(TRAIN_WOE) $(TEST_PRED) $(SRC)/step9c_calibration.py
	$(PYTHON) $(SRC)/step9c_calibration.py
$(CAL_LR): $(CAL_POST) ;

# Step 10: LGD
$(LOANS_LGD): $(LOANS_MACROS) $(CAL_POST) $(SRC)/step10_lgd_estimation.py
	$(PYTHON) $(SRC)/step10_lgd_estimation.py

# Step 11: EAD via contractual amortization
$(LOANS_EAD): $(LOANS_LGD) $(SRC)/step11_ead_projection.py
	$(PYTHON) $(SRC)/step11_ead_projection.py

# Step 12: ECL combination (primary: ECL_HEADLINE)
$(ECL_HEADLINE): \
		$(LOANS_EAD) $(BINNING_JSON) $(TRAIN_WOE) $(TEST_PRED) \
		$(SRC)/step12_ecl_combination.py
	$(PYTHON) $(SRC)/step12_ecl_combination.py
$(LOANS_ECL): $(ECL_HEADLINE) ;

# Step 13: data-driven macro overlay (primary: OVERLAY_JSON)
$(OVERLAY_JSON): \
		$(LOANS_ECL) $(TRAIN_WOE) $(TEST_WOE) $(BINNING_JSON) \
		$(SRC)/step13_macro_overlay.py
	$(PYTHON) $(SRC)/step13_macro_overlay.py
$(LOANS_OVERLAY): $(OVERLAY_JSON) ;

# Step 14: validation pack + regulatory overlay
$(REG_OVERLAY): $(LOANS_OVERLAY) $(TEST_PRED) $(SRC)/step14_validation.py
	$(PYTHON) $(SRC)/step14_validation.py

# Step 15: dashboard data
$(DASH_SUMMARY): $(LOANS_OVERLAY) $(ECL_HEADLINE) $(OVERLAY_JSON) $(REG_OVERLAY) \
		$(SRC)/step15_dashboard_data.py
	$(PYTHON) $(SRC)/step15_dashboard_data.py

# Documentation
$(SUMMARY_MD): $(REG_OVERLAY) $(DASH_SUMMARY) $(SRC)/build_project_summary.py
	$(PYTHON) $(SRC)/build_project_summary.py

$(DOSSIER_MD): $(SUMMARY_MD) $(LOANS_OVERLAY) $(SRC)/build_final_dossier.py
	$(PYTHON) $(SRC)/build_final_dossier.py

# ---------- QA ----------

validate: $(REG_OVERLAY) $(SUMMARY_MD) $(DOSSIER_MD)  ## Run validator (Steps 7–18)
	$(PYTHON) $(SRC)/validate_pipeline_steps_7_8.py

audit: $(REG_OVERLAY) $(SUMMARY_MD) $(DOSSIER_MD)     ## Run full pipeline audit
	$(PYTHON) $(SRC)/audit_full_pipeline.py

test:  ## Run pytest unit tests for math primitives
	$(PYTEST) -q tests/

qa: test validate audit  ## test + validate + audit (rebuilds stale artifacts first)

qa-fast:  ## Run QA without rebuilding artifacts (assumes pipeline is current)
	$(PYTEST) -q tests/
	$(PYTHON) $(SRC)/validate_pipeline_steps_7_8.py
	$(PYTHON) $(SRC)/audit_full_pipeline.py

dag:  ## Print the build DAG that make would execute (dry run)
	@$(MAKE) -n all | head -40

# ---------- Setup / cleanup ----------

init:  ## Create venv and install dependencies (one-time setup)
	python3 -m venv .venv
	.venv/bin/pip install -U pip
	.venv/bin/pip install -r requirements.txt

clean-docs:  ## Remove generated docs (summary, dossier, audit reports)
	@rm -fv $(SUMMARY_MD) $(DOSSIER_MD)
	@rm -fv $(DOCS)/audit_report_full_*.md
	@rm -fv $(DOCS)/validation_report.md

clean:  ## Remove ALL generated artifacts (interactive, asks first)
	@printf "About to delete every artifact under data/, models/, and generated docs/. Continue? [y/N] "; \
	read ans; [ "$$ans" = "y" ] || { echo "aborted"; exit 1; }
	rm -rf $(DATA)/*.parquet $(DASHBOARD) $(MODELS)/*.pkl
	rm -fv $(DOCS)/ecl_*.json $(DOCS)/ecl_*.csv
	rm -fv $(DOCS)/overlay_*.json $(DOCS)/overlay_*.csv
	rm -fv $(DOCS)/validation_*.json $(DOCS)/validation_*.csv
	rm -fv $(DOCS)/calibration_*.json $(DOCS)/binning_summary.json
	rm -fv $(DOCS)/feature_classification.json $(DOCS)/lgd_*.* $(DOCS)/ead_*.*
	rm -fv $(DOCS)/model_evaluation.json $(DOCS)/calibrators.json
	rm -fv $(SUMMARY_MD) $(DOSSIER_MD)
	rm -fv $(DOCS)/audit_report_full_*.md $(DOCS)/validation_report.md
