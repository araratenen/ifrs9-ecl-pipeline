"""Shared fixtures for math-primitive unit tests.

Tests in this directory verify the closed-form math used by the pipeline in
isolation (independent of pandas/sklearn integration). Fixtures here load
real artifacts to spot-check that the production code respects the same
formulas.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def docs_dir(project_root: Path) -> Path:
    return project_root / "docs"


@pytest.fixture(scope="session")
def models_dir(project_root: Path) -> Path:
    return project_root / "models"


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    return project_root / "data"


@pytest.fixture(scope="session")
def ecl_headline(docs_dir: Path) -> dict:
    return json.loads((docs_dir / "ecl_headline.json").read_text())


@pytest.fixture(scope="session")
def overlay_headline(docs_dir: Path) -> dict:
    return json.loads((docs_dir / "ecl_overlay_headline.json").read_text())


@pytest.fixture(scope="session")
def regulatory_overlay(docs_dir: Path) -> dict:
    return json.loads((docs_dir / "validation_regulatory_overlay.json").read_text())


@pytest.fixture(scope="session")
def lr_calibrator(models_dir: Path):
    """Loaded sklearn LogisticRegression Platt scaler."""
    import joblib
    return joblib.load(models_dir / "pd_logistic_calibrator.pkl")


@pytest.fixture(scope="session")
def gbm_calibrator(models_dir: Path):
    import joblib
    return joblib.load(models_dir / "pd_xgboost_calibrator.pkl")


@pytest.fixture(scope="session")
def sample_active_loan(data_dir: Path):
    """One active loan from loans_with_ead.parquet for end-to-end checks."""
    import pandas as pd
    ead = pd.read_parquet(data_dir / "loans_with_ead.parquet")
    active = ead[(ead["months_remaining"] > 0) & (ead["ead_12m"] > 0)]
    if active.empty:
        pytest.skip("no active loans in loans_with_ead.parquet")
    return active.iloc[0].to_dict()
