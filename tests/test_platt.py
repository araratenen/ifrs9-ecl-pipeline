"""Unit tests for the Platt scaler (Step 9c).

Platt scaling for binary classifiers fits a 1-d logistic regression on the
uncalibrated scores:

    p_calibrated(x) = sigmoid(intercept + coef * x)

The calibrators are saved as sklearn LogisticRegression objects.
"""
import numpy as np
import pandas as pd
import pytest


def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + np.exp(-z))


def platt_predict(intercept: float, coef: float, score: float) -> float:
    return sigmoid(intercept + coef * score)


# ---------- Sigmoid sanity ----------

def test_sigmoid_at_zero_is_one_half():
    assert sigmoid(0) == pytest.approx(0.5)


def test_sigmoid_bounded():
    for z in [-100, -10, -1, 0, 1, 10, 100]:
        v = sigmoid(z)
        assert 0 <= v <= 1


def test_sigmoid_monotone():
    grid = np.linspace(-10, 10, 100)
    vals = sigmoid(grid)
    assert (np.diff(vals) > 0).all()


# ---------- Loaded calibrator ----------

def test_lr_calibrator_is_logistic_regression(lr_calibrator):
    from sklearn.linear_model import LogisticRegression
    assert isinstance(lr_calibrator, LogisticRegression)


def test_lr_calibrator_has_one_input_feature(lr_calibrator):
    """Platt scales a single uncalibrated score, so coef must be shape (1, 1)."""
    assert lr_calibrator.coef_.shape == (1, 1)
    assert lr_calibrator.intercept_.shape == (1,)


def test_lr_calibrator_predict_proba_matches_closed_form(lr_calibrator):
    """sklearn predict_proba(X)[:, 1] must equal sigmoid(intercept + coef*x)."""
    intercept = float(lr_calibrator.intercept_[0])
    coef = float(lr_calibrator.coef_[0, 0])

    grid = np.array([[0.0], [0.05], [0.10], [0.25], [0.5], [0.75], [0.99]])
    sklearn_pred = lr_calibrator.predict_proba(grid)[:, 1]
    closed_form = sigmoid(intercept + coef * grid.ravel())

    np.testing.assert_allclose(sklearn_pred, closed_form, rtol=1e-12, atol=1e-12)


def test_lr_calibrator_is_monotone(lr_calibrator):
    """Higher uncalibrated score → higher calibrated probability (positive coef)."""
    grid = np.linspace(0.0, 1.0, 50).reshape(-1, 1)
    pred = lr_calibrator.predict_proba(grid)[:, 1]
    assert (np.diff(pred) > 0).all(), "Platt-calibrated PD must be monotone in score"


def test_lr_calibrator_output_in_unit_interval(lr_calibrator):
    grid = np.linspace(0.0, 1.0, 100).reshape(-1, 1)
    pred = lr_calibrator.predict_proba(grid)[:, 1]
    assert (pred >= 0).all() and (pred <= 1).all()


def test_gbm_calibrator_same_signature(gbm_calibrator):
    """The GBM calibrator must satisfy the same shape contract as the LR one."""
    assert gbm_calibrator.coef_.shape == (1, 1)
    assert gbm_calibrator.intercept_.shape == (1,)


# ---------- Reconcile against pipeline test predictions ----------

def test_test_predictions_pd_lr_calibrated_matches_calibrator(
    data_dir, lr_calibrator,
):
    """The `pd_lr_calibrated` column in test_predictions must equal what the
    saved LR calibrator predicts on the uncalibrated `pd_lr_uncalibrated` column."""
    df = pd.read_parquet(data_dir / "test_predictions.parquet")
    if "pd_lr" not in df.columns or "pd_lr_calibrated" not in df.columns:
        pytest.skip("uncalibrated/calibrated columns not in test_predictions")

    sample = df.sample(200, random_state=0)
    X = sample["pd_lr"].to_numpy().reshape(-1, 1)
    expected = lr_calibrator.predict_proba(X)[:, 1]
    actual = sample["pd_lr_calibrated"].to_numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
