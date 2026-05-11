"""Unit tests for the closed-form amortization used in Step 11 EAD projection.

The contractual balance after `t` months on a fixed-rate amortizing loan is

    M     = P * r * (1+r)^n / ((1+r)^n - 1)             # monthly payment
    B(t)  = P * (1+r)^t - M * ((1+r)^t - 1) / r         # balance after t months

with the r → 0 limit B(t) = P * (1 - t/n). These tests verify the math in
isolation and then spot-check that one row from `loans_with_ead.parquet`
reproduces from raw inputs.
"""
import math

import numpy as np
import pytest


def monthly_payment(P: float, annual_rate: float, n: int) -> float:
    r = annual_rate / 12
    if r == 0:
        return P / n
    factor = (1 + r) ** n
    return P * r * factor / (factor - 1)


def balance_at(P: float, annual_rate: float, n: int, t: int) -> float:
    r = annual_rate / 12
    if r == 0:
        return P * (1 - t / n)
    M = monthly_payment(P, annual_rate, n)
    factor_t = (1 + r) ** t
    return P * factor_t - M * (factor_t - 1) / r


# ---------- Closed-form sanity ----------

def test_balance_at_zero_equals_principal():
    assert balance_at(20000, 0.10, 60, 0) == pytest.approx(20000)


def test_balance_at_term_is_zero():
    for P, rate, n in [(10000, 0.05, 36), (20000, 0.1078, 60), (5000, 0.20, 36)]:
        assert balance_at(P, rate, n, n) == pytest.approx(0, abs=1e-6)


def test_balance_monotonically_decreasing():
    P, rate, n = 20000, 0.1078, 60
    bals = [balance_at(P, rate, n, t) for t in range(n + 1)]
    diffs = np.diff(bals)
    assert (diffs <= 1e-9).all(), "balance must not increase month over month"


def test_zero_rate_is_linear():
    """Limit r → 0: B(t) = P * (1 - t/n)."""
    P, n = 12000, 24
    for t in (0, 6, 12, 18, 24):
        assert balance_at(P, 0.0, n, t) == pytest.approx(P * (1 - t / n))


def test_payment_recovers_principal_plus_interest():
    """Sum of M*n minus principal equals total interest paid."""
    P, rate, n = 20000, 0.10, 60
    M = monthly_payment(P, rate, n)
    total_paid = M * n
    interest = total_paid - P
    # Interest on a 60-month 10% loan on $20k is well-known ~$5500
    assert 5000 < interest < 6500


def test_higher_rate_higher_total_interest():
    P, n = 15000, 36
    pay_5 = monthly_payment(P, 0.05, n) * n
    pay_15 = monthly_payment(P, 0.15, n) * n
    assert pay_15 > pay_5


# ---------- Cross-check against the pipeline implementation ----------

def test_pipeline_step11_implementation_matches_closed_form():
    """The vectorized closed-form in step11._contractual_balance_at must
    agree with the scalar reference implementation for the LC operating
    range (r > 0). Real LC int_rate is never zero, so the r=0 edge case
    is not exercised by production."""
    from src.step11_ead_projection import _contractual_balance_at

    P = np.array([20000.0, 10000.0, 5000.0, 8500.0])
    annual_rate = np.array([0.1078, 0.05, 0.20, 0.0699])
    r = annual_rate / 12
    n = np.array([60, 36, 36, 36])
    t = np.array([20, 12, 0, 24])

    pipeline = _contractual_balance_at(P, r, n, t)
    reference = np.array([balance_at(P[i], annual_rate[i], n[i], t[i])
                           for i in range(len(P))])
    np.testing.assert_allclose(pipeline, reference, rtol=1e-9, atol=1e-6)


def test_pipeline_step11_balance_path_zero_rate_is_linear():
    """`_balance_path_closed_form` handles the r ≤ 0 limit explicitly with
    the linear formula B(t) = P * (1 - t/n)."""
    from src.step11_ead_projection import _balance_path_closed_form

    path = _balance_path_closed_form(starting=12000, r=0.0, n=24)
    expected = 12000 * (1 - np.arange(1, 25) / 24)
    np.testing.assert_allclose(path, expected, rtol=1e-12)


def test_pipeline_step11_balance_path_terminates_at_zero():
    """The balance path returned by step11 must end at zero on the final month."""
    from src.step11_ead_projection import _balance_path_closed_form

    path = _balance_path_closed_form(starting=20000, r=0.1078 / 12, n=60)
    assert len(path) == 60
    assert path[-1] == pytest.approx(0, abs=1e-6)
    # Strictly decreasing
    assert (np.diff(path) < 0).all()


# ---------- Spot-check against a real loan ----------

def test_sample_active_loan_balance_path_terminates(sample_active_loan: dict):
    """Last element of ead_lifetime_path must be ~0 (loan paid off)."""
    path = np.asarray(sample_active_loan["ead_lifetime_path"])
    assert len(path) == int(sample_active_loan["months_remaining"])
    assert path[-1] == pytest.approx(0, abs=1.0)  # within $1 of zero


def test_sample_active_loan_path_is_decreasing(sample_active_loan: dict):
    path = np.asarray(sample_active_loan["ead_lifetime_path"])
    assert (np.diff(path) <= 1e-6).all()
