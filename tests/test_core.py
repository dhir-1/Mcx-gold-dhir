"""
tests/test_core.py - Unit tests for core calculation logic.
Tests all five required validation rules using pytest.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import pandas as pd
import numpy as np
from core import (
    convert_to_999_equivalent,
    calculate_blend_weights,
    calculate_gap_pct,
    calculate_normal_level,
    determine_signal,
    compute_calendar_timeline,
    CONTRACT_SPECS,
    DEFAULT_BREAK_EVEN_PCT,
    DEFAULT_EXPIRY_CUTOFF_DAYS,
)


# -----------------------------------------------------------------------------
# TEST (a): GOLDM conversion to 999-equivalent on a known number
# Plain-English: 10,000 Rs/g for 995 gold must equal 10,000 * 999 / 995
# = 10,040.2010... Rs/g in 999 gold terms. Other products must remain unchanged.
# -----------------------------------------------------------------------------
def test_goldm_purity_conversion():
    known_close_per_gram = 10000.0
    expected_goldm_999 = 10000.0 * (999.0 / 995.0)

    # GOLDM must be scaled by 999/995
    res_goldm = convert_to_999_equivalent("GOLDM", known_close_per_gram)
    assert pytest.approx(res_goldm, rel=1e-6) == expected_goldm_999
    assert pytest.approx(res_goldm, rel=1e-4) == 10040.2010

    # 999 contracts must return identical price
    for sym in ["GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]:
        assert convert_to_999_equivalent(sym, known_close_per_gram) == known_close_per_gram


# -----------------------------------------------------------------------------
# TEST (b): Blend weights for E=2026-10-30, E1=2026-10-05, E2=2026-11-05
# Plain-English: Calendar days (E2 - E1) = 31, (E - E1) = 25.
# w2 = 25/31 = 0.8064516..., w1 = 6/31 = 0.1935483...
# -----------------------------------------------------------------------------
def test_blend_weights():
    e = pd.Timestamp("2026-10-30")
    e1 = pd.Timestamp("2026-10-05")
    e2 = pd.Timestamp("2026-11-05")

    w1, w2 = calculate_blend_weights(e, e1, e2)

    assert pytest.approx(w1, abs=1e-5) == 0.193548
    assert pytest.approx(w2, abs=1e-5) == 0.806452
    assert pytest.approx(w1 + w2, abs=1e-9) == 1.0


# -----------------------------------------------------------------------------
# TEST (c): Gap formula on a small hand-made table
# Plain-English: gap_pct = (price / reference - 1) * 100.
# -----------------------------------------------------------------------------
def test_gap_formula():
    # Hand-made values
    prices = [100.0, 101.0, 99.5, 105.0]
    refs = [100.0, 100.0, 100.0, 100.0]
    expected_gaps = [0.0, 1.0, -0.5, 5.0]

    for p, r, expected in zip(prices, refs, expected_gaps):
        gap = calculate_gap_pct(p, r)
        assert pytest.approx(gap, rel=1e-6) == expected


# -----------------------------------------------------------------------------
# TEST (d): Quiet case where excess is below break-even returns "No signal"
# Plain-English: If gap is 0.40%, normal is 0.30% (excess = 0.10% <= break-even 0.22%),
# the system must stay quiet and return "No signal".
# -----------------------------------------------------------------------------
def test_quiet_case_no_signal():
    gap = 0.40
    normal = 0.30
    break_even = 0.22
    days_to_expiry = 25

    code, msg = determine_signal(gap, normal, break_even_pct=break_even, days_to_expiry=days_to_expiry)
    assert code == "NO_SIGNAL"
    assert msg == "No signal"

    # Also test when days_to_expiry <= 10 (cutoff rule)
    large_excess_gap = 1.00  # excess = 0.70% > 0.22%
    code_near_exp, msg_near_exp = determine_signal(
        large_excess_gap, normal, break_even_pct=break_even, days_to_expiry=5, cutoff_days=10
    )
    assert code_near_exp == "NO_SIGNAL"
    assert msg_near_exp == "No signal"

    # Test true signal case
    code_sig, msg_sig = determine_signal(
        large_excess_gap, normal, break_even_pct=break_even, days_to_expiry=25, cutoff_days=10
    )
    assert code_sig == "SIGNAL"
    assert "Signal:" in msg_sig


# -----------------------------------------------------------------------------
# TEST (e): Normal level uses only past days (no look-ahead bias)
# Plain-English: Changing today's value must NOT change today's normal level.
# -----------------------------------------------------------------------------
def test_normal_level_no_lookahead():
    # Past 5 days of gaps
    base_gaps = [0.10, 0.20, 0.30, 0.40, 0.50]
    series_original = pd.Series(base_gaps + [0.35])   # Today's value = 0.35
    series_modified = pd.Series(base_gaps + [99.99])  # Changed today's value = 99.99

    norm_orig = calculate_normal_level(series_original, window=5)
    norm_mod = calculate_normal_level(series_modified, window=5)

    # Today is index 5 (the 6th element)
    # The normal level for today must strictly depend on past days (index 0..4)
    # Changing index 5 from 0.35 to 99.99 MUST NOT change today's normal level!
    assert norm_orig.iloc[5] == norm_mod.iloc[5]
    expected_past_mean = np.mean(base_gaps)
    assert pytest.approx(norm_orig.iloc[5], rel=1e-6) == expected_past_mean


# -----------------------------------------------------------------------------
# TEST 1 (New): Tender start for running contract expiring 2026-10-30 with data ending 2026-10-01
# Plain-English: Counting back 3 business days from 2026-10-30 yields 2026-10-28
# (the 28th, 29th and 30th, all weekdays), with tender_is_estimated = True.
# -----------------------------------------------------------------------------
def test_tender_period_running_contract():
    dates = pd.date_range("2026-09-01", "2026-10-01", freq="B")
    df_sample = pd.DataFrame({
        "symbol": "GOLDTEN",
        "expiry_date": pd.Timestamp("2026-10-30"),
        "date": dates,
        "volume_lots": 100,
        "traded": True,
    })
    contracts_info, _ = compute_calendar_timeline(df_sample)
    assert len(contracts_info) == 1
    c = contracts_info[0]
    assert c["tender_start"] == pd.Timestamp("2026-10-28")
    assert c["tender_is_estimated"] is True


# -----------------------------------------------------------------------------
# TEST 2 (New): Tender start for already expired contract uses last 3 dates from data
# Plain-English: A finished contract (data ends on expiry date) uses the 3rd to last
# actual trading date from the dataset, with tender_is_estimated = False.
# -----------------------------------------------------------------------------
def test_tender_period_expired_contract():
    # Data dates ending on expiry date 2026-08-31
    dates = [
        pd.Timestamp("2026-08-25"),
        pd.Timestamp("2026-08-26"),
        pd.Timestamp("2026-08-27"),
        pd.Timestamp("2026-08-28"),
        pd.Timestamp("2026-08-31"),
    ]
    df_sample = pd.DataFrame({
        "symbol": "GOLDTEN",
        "expiry_date": pd.Timestamp("2026-08-31"),
        "date": dates,
        "volume_lots": 100,
        "traded": True,
    })
    contracts_info, _ = compute_calendar_timeline(df_sample)
    assert len(contracts_info) == 1
    c = contracts_info[0]
    # 3rd from last date is 2026-08-27
    assert c["tender_start"] == pd.Timestamp("2026-08-27")
    assert c["tender_is_estimated"] is False
