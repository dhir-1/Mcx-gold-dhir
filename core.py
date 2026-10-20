"""
core.py - Core calculation logic and MCX contract specifications.
This module contains pure Python/pandas/numpy calculations.
No Streamlit code is included in this module.
"""

import os
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
# CONTRACT SPECIFICATIONS (Source: MCX Contract Specifications)
# All products use parameters from official MCX specifications.
# -----------------------------------------------------------------------------
CONTRACT_SPECS: Dict[str, Dict[str, Any]] = {
    "GOLDM": {
        "grams_per_lot": 100,
        "quote_unit_grams": 10,
        "purity": 995,
        "quote_basis": "Ex-Ahmedabad",
        "expiry_rule_text": "5th of expiry month",
        "tender_days": 3,
        "tender_margin_pct": 8.34,
        "delivery_centre": "Ahmedabad (additional: Mumbai, New Delhi)",
        "making_charge_rs": "None stated in spec",
        "final_settlement_method": "average of last polled spot prices of the last 3 trading days",
    },
    "GOLDTEN": {
        "grams_per_lot": 10,
        "quote_unit_grams": 10,
        "purity": 999,
        "quote_basis": "Ex-Ahmedabad",
        "expiry_rule_text": "last calendar day of expiry month",
        "tender_days": 3,
        "tender_margin_pct": 8.34,
        "delivery_centre": "Ahmedabad",
        "making_charge_rs": "300 per 10 g",
        "final_settlement_method": "spot price polled on expiry day (DDR)",
    },
    "GOLDGUINEA": {
        "grams_per_lot": 8,
        "quote_unit_grams": 8,
        "purity": 999,
        "quote_basis": "Ex-Ahmedabad",
        "expiry_rule_text": "last calendar day of expiry month",
        "tender_days": 3,
        "tender_margin_pct": 8.34,
        "delivery_centre": "Ahmedabad",
        "making_charge_rs": "200 per 8 g",
        "final_settlement_method": "DDR",
    },
    "GOLDPETAL": {
        "grams_per_lot": 1,
        "quote_unit_grams": 1,
        "purity": 999,
        "quote_basis": "Ex-Mumbai",
        "expiry_rule_text": "last calendar day of expiry month",
        "tender_days": 3,
        "tender_margin_pct": 8.34,
        "delivery_centre": "Mumbai",
        "making_charge_rs": "100 per 1 g",
        "final_settlement_method": "DDR",
    },
}

# -----------------------------------------------------------------------------
# DEFAULT PARAMETER CONSTANTS
# -----------------------------------------------------------------------------
DEFAULT_BREAK_EVEN_PCT: float = 0.22
DEFAULT_ROLLING_WINDOW: int = 10
DEFAULT_EXPIRY_CUTOFF_DAYS: int = 10
DESIGN_PERIOD_CUTOFF: str = "2026-09-11"
UNSEEN_PERIOD_START: str = "2026-09-14"

# Palette: Gold accent with dark blue slate text
COLOR_ACCENT_GOLD: str = "#C98A1B"
COLOR_TEXT_NAVY: str = "#2F4558"


# -----------------------------------------------------------------------------
# RULE 2: PURITY ADJUSTMENT
# Plain-English: GOLDM is quoted at 995 purity. The MCX physical conversion factor
# to 999 gold is 999 / 995. GOLDTEN, GOLDGUINEA, and GOLDPETAL are already 999 purity.
# -----------------------------------------------------------------------------
def convert_to_999_equivalent(symbol: str, close_per_gram: float) -> float:
    """Convert price per gram to 999-equivalent terms."""
    if symbol == "GOLDM":
        return close_per_gram * (999.0 / 995.0)
    return float(close_per_gram)


# -----------------------------------------------------------------------------
# RULE 3: EXPIRY BLEND WEIGHTS
# Plain-English: Linear interpolation between surrounding GOLDM expiries (E1 and E2)
# based on calendar days to match the target 999 expiry E.
# -----------------------------------------------------------------------------
def calculate_blend_weights(
    e: pd.Timestamp, e1: pd.Timestamp, e2: pd.Timestamp
) -> Tuple[float, float]:
    """Calculate calendar-day interpolation weights (w1, w2) for surrounding expiries."""
    total_days = (e2 - e1).days
    if total_days <= 0:
        raise ValueError(f"Invalid expiry sequence: E1 ({e1}) must precede E2 ({e2})")
    w2 = (e - e1).days / total_days
    w1 = 1.0 - w2
    return w1, w2


# -----------------------------------------------------------------------------
# RULE 4: GAP FORMULA
# Plain-English: Percentage price gap = (price / reference - 1) * 100.
# -----------------------------------------------------------------------------
def calculate_gap_pct(price: float, reference: float) -> float:
    """Calculate percentage price gap relative to a reference price."""
    if reference == 0 or np.isnan(reference) or np.isnan(price):
        return np.nan
    return float((price / reference - 1.0) * 100.0)


# -----------------------------------------------------------------------------
# RULE 6: NORMAL LEVEL CALCULATION
# Plain-English: Rolling mean of the gap over previous N trading days of the same expiry.
# Excludes today (uses shift(1)) so there is no look-ahead bias.
# -----------------------------------------------------------------------------
def calculate_normal_level(
    gap_series: pd.Series, window: int = DEFAULT_ROLLING_WINDOW
) -> pd.Series:
    """Compute 10-day rolling mean of past days only (exclude today)."""
    return gap_series.shift(1).rolling(window=window, min_periods=1).mean()


# -----------------------------------------------------------------------------
# RULE 7: SIGNAL RULE (Quiet by default)
# Plain-English: A signal triggers only if |gap - normal_level| > break_even_pct
# AND days_to_expiry > cutoff_days. Otherwise returns 'No signal'.
# -----------------------------------------------------------------------------
def determine_signal(
    gap: float,
    normal_level: float,
    break_even_pct: float = DEFAULT_BREAK_EVEN_PCT,
    days_to_expiry: int = 15,
    cutoff_days: int = DEFAULT_EXPIRY_CUTOFF_DAYS,
) -> Tuple[str, str]:
    """
    Evaluate trading signal.
    Returns:
        (code, message): ('SIGNAL', 'Signal: ...') or ('NO_SIGNAL', 'No signal')
    """
    if pd.isna(gap) or pd.isna(normal_level):
        return "NO_SIGNAL", "No signal"

    excess = gap - normal_level
    if abs(excess) > break_even_pct and days_to_expiry > cutoff_days:
        direction = "above" if excess > 0 else "below"
        msg = f"Signal: {abs(excess):.2f}% {direction} normal level"
        return "SIGNAL", msg

    return "NO_SIGNAL", "No signal"


# -----------------------------------------------------------------------------
# DATA LOADERS (Safe, read-only, non-crashing)
# -----------------------------------------------------------------------------
def _resolve_data_path(filepath: str) -> str:
    """Resolve file path relative to repo root if not found in current directory."""
    if os.path.exists(filepath):
        return filepath
    alt = os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    if os.path.exists(alt):
        return alt
    return filepath


def load_clean_data(filepath: str = "out/gold_futures_clean.csv") -> pd.DataFrame:
    """Load cleaned gold futures data with date parsing and traded filtering."""
    filepath = _resolve_data_path(filepath)
    if not os.path.exists(filepath):
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    df["date"] = pd.to_datetime(df["date"])
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])

    # Rule 1: Use only traded rows
    if "traded" in df.columns:
        df = df[df["traded"] == True].copy()
    elif "volume_lots" in df.columns:
        df = df[df["volume_lots"] > 0].copy()

    # Rule 2: Add 999-equivalent price per gram
    df["price_999_per_gram"] = df.apply(
        lambda r: convert_to_999_equivalent(r["symbol"], r["close_per_gram"]),
        axis=1,
    )

    # Volume in physical grams
    df["grams_per_lot"] = df["symbol"].map(
        lambda s: CONTRACT_SPECS.get(s, {}).get("grams_per_lot", 1)
    )
    df["volume_grams"] = df["volume_lots"] * df["grams_per_lot"]

    return df.sort_values(["symbol", "expiry_date", "date"]).reset_index(drop=True)


def load_ddr_data(filepath: str = "results_ddr/ddr.csv") -> pd.DataFrame:
    """Load MCX Due Date Rate final settlement reference table."""
    filepath = _resolve_data_path(filepath)
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    df["expiry_date"] = pd.to_datetime(df["expiry_date"])
    df["ddr_per_gram"] = df["ddr_price"] / df["unit_grams"]
    return df


def load_trades_data(filepath: str = "results_bt/trades.csv") -> pd.DataFrame:
    """Load backtest trade records."""
    filepath = _resolve_data_path(filepath)
    if not os.path.exists(filepath):
        return pd.DataFrame()
    df = pd.read_csv(filepath)
    if "entry_date" in df.columns:
        df["entry_date"] = pd.to_datetime(df["entry_date"])
    if "exit_date" in df.columns:
        df["exit_date"] = pd.to_datetime(df["exit_date"])
    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"])
    return df


# -----------------------------------------------------------------------------
# RULE 8: REPLAY / AS-OF FILTERING
# Plain-English: Filter all dataset records to date <= as_of_date before computing.
# -----------------------------------------------------------------------------
def filter_data_as_of(df: pd.DataFrame, as_of_date: pd.Timestamp) -> pd.DataFrame:
    """Filter records to simulate historical as-of state (no look-ahead)."""
    if df.empty:
        return df
    return df[df["date"] <= as_of_date].copy()


# -----------------------------------------------------------------------------
# RULE 3 & 4: COMPUTE REFERENCE & GAPS FOR EXPIRY
# -----------------------------------------------------------------------------
def compute_expiry_gaps(
    df: pd.DataFrame,
    target_expiry: pd.Timestamp,
    as_of_date: Optional[pd.Timestamp] = None,
    window: int = DEFAULT_ROLLING_WINDOW,
    break_even_pct: float = DEFAULT_BREAK_EVEN_PCT,
    cutoff_days: int = DEFAULT_EXPIRY_CUTOFF_DAYS,
) -> Tuple[pd.DataFrame, str]:
    """
    Compute daily reference price, gaps, normal levels, and signals for a 999 expiry.
    Returns:
        (df_gaps, reference_description)
    """
    if df.empty:
        return pd.DataFrame(), "No data loaded"

    if as_of_date is not None:
        df_work = filter_data_as_of(df, as_of_date)
    else:
        df_work = df.copy()

    # Identify contracts for target expiry
    df_exp = df_work[df_work["expiry_date"] == target_expiry]
    if df_exp.empty:
        return pd.DataFrame(), f"No data for expiry {target_expiry.strftime('%Y-%m-%d')}"

    # Check for surrounding GOLDM contracts
    goldm_df = df_work[df_work["symbol"] == "GOLDM"]
    goldm_expiries = sorted(goldm_df["expiry_date"].unique())

    goldm_e1_list = [g for g in goldm_expiries if g <= target_expiry]
    goldm_e2_list = [g for g in goldm_expiries if g > target_expiry]

    use_blend = False
    ref_desc = ""
    e1, e2 = None, None
    w1, w2 = 0.0, 0.0

    if goldm_e1_list and goldm_e2_list:
        e1 = max(goldm_e1_list)
        e2 = min(goldm_e2_list)
        try:
            w1, w2 = calculate_blend_weights(target_expiry, e1, e2)
            use_blend = True
            ref_desc = f"GOLDM blend ({e1.strftime('%d%b%y')} & {e2.strftime('%d%b%y')})"
        except ValueError:
            use_blend = False

    if not use_blend:
        ref_desc = "reference: GOLDTEN (GOLDM contracts for this expiry not loaded)"

    # Pivot 999 contracts of target expiry
    pivot_prices = df_exp.pivot(index="date", columns="symbol", values="price_999_per_gram")
    pivot_days = df_exp.pivot(index="date", columns="symbol", values="days_to_expiry")

    # Build reference series and track which reference was actually used per date
    dates = sorted(pivot_prices.index)
    ref_series = pd.Series(index=dates, dtype=float)
    ref_used_map = {}

    if use_blend and e1 is not None and e2 is not None:
        m1_series = goldm_df[goldm_df["expiry_date"] == e1].set_index("date")["price_999_per_gram"]
        m2_series = goldm_df[goldm_df["expiry_date"] == e2].set_index("date")["price_999_per_gram"]

        for d in dates:
            days_left_e1 = (e1 - d).days
            # Must both trade and E1 has > 7 calendar days left
            if d in m1_series.index and d in m2_series.index and days_left_e1 > 7:
                ref_series[d] = w1 * m1_series.loc[d] + w2 * m2_series.loc[d]
                ref_used_map[d] = "GOLDM blend"
            else:
                # If blend is not available on this date, fallback to GOLDTEN if present
                if "GOLDTEN" in pivot_prices.columns and not pd.isna(pivot_prices.loc[d, "GOLDTEN"]):
                    ref_series[d] = pivot_prices.loc[d, "GOLDTEN"]
                    ref_used_map[d] = "GOLDTEN (fallback)"
                else:
                    ref_used_map[d] = "None"
    else:
        if "GOLDTEN" in pivot_prices.columns:
            ref_series = pivot_prices["GOLDTEN"]
            for d in dates:
                ref_used_map[d] = "GOLDTEN (fallback)"

    # Assemble gap dataframe
    rows = []
    contracts = [s for s in ["GOLDTEN", "GOLDGUINEA", "GOLDPETAL"] if s in pivot_prices.columns]

    for d in dates:
        ref_p = ref_series.get(d, np.nan)
        ref_used_today = ref_used_map.get(d, "GOLDTEN (fallback)" if not use_blend else "GOLDM blend")

        for sym in contracts:
            # Rule 4: Skip contract compared with itself
            if ref_used_today == "GOLDTEN (fallback)" and sym == "GOLDTEN":
                continue

            price = pivot_prices.loc[d, sym] if sym in pivot_prices.columns else np.nan
            dte = (
                pivot_days.loc[d, sym]
                if sym in pivot_days.columns and not pd.isna(pivot_days.loc[d, sym])
                else (target_expiry - d).days
            )
            gap = calculate_gap_pct(price, ref_p) if not pd.isna(ref_p) else np.nan

            rows.append({
                "date": d,
                "expiry_date": target_expiry,
                "symbol": sym,
                "price_per_gram": price,
                "reference_price": ref_p,
                "reference_desc": ref_desc,
                "reference_used": ref_used_today,
                "gap_pct": gap,
                "days_to_expiry": int(dte),
            })

    if not rows:
        return pd.DataFrame(), ref_desc

    res_df = pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)

    # Calculate rolling normal level, excess, and signal per symbol
    res_df["normal_level"] = np.nan
    res_df["excess"] = np.nan
    res_df["signal_code"] = "NO_SIGNAL"
    res_df["status"] = "No signal"

    for sym in res_df["symbol"].unique():
        idx = res_df["symbol"] == sym
        sub = res_df.loc[idx].sort_values("date")
        norm = calculate_normal_level(sub["gap_pct"], window=window)
        res_df.loc[sub.index, "normal_level"] = norm
        excess = sub["gap_pct"] - norm
        res_df.loc[sub.index, "excess"] = excess

        for row_i in sub.index:
            g = res_df.loc[row_i, "gap_pct"]
            nl = res_df.loc[row_i, "normal_level"]
            dte = res_df.loc[row_i, "days_to_expiry"]
            code, stat = determine_signal(g, nl, break_even_pct, dte, cutoff_days)
            res_df.loc[row_i, "signal_code"] = code
            res_df.loc[row_i, "status"] = stat

    return res_df, ref_desc


# -----------------------------------------------------------------------------
# TAB 1: TODAY / AS-OF SUMMARY
# -----------------------------------------------------------------------------
def get_today_status_and_tables(
    df: pd.DataFrame,
    as_of_date: pd.Timestamp,
    window: int = DEFAULT_ROLLING_WINDOW,
    break_even_pct: float = DEFAULT_BREAK_EVEN_PCT,
    cutoff_days: int = DEFAULT_EXPIRY_CUTOFF_DAYS,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Produce banner status and summary tables for active expiries as of chosen date.
    Active expiry rule: expiry_date >= as_of_date.
    """
    if df.empty:
        return {"code": "NO_SIGNAL", "message": "No data loaded"}, []

    # If the chosen as-of date has no trading data, return NO_DATA
    if df[df["date"] == as_of_date].empty:
        return {"code": "NO_DATA", "message": "No trading data for this date", "count": 0}, []

    # Active expiries: expiry_date >= as_of_date
    df_asof = filter_data_as_of(df, as_of_date)
    active_expiries = sorted(
        df_asof[df_asof["expiry_date"] >= as_of_date]["expiry_date"].unique()
    )

    # Focus on 999 expiries
    exp_999 = [
        e for e in active_expiries
        if set(df_asof[df_asof["expiry_date"] == e]["symbol"]).intersection(
            {"GOLDTEN", "GOLDGUINEA", "GOLDPETAL"}
        )
    ]

    all_signals = []
    tables_per_expiry = []

    for exp in exp_999:
        gaps_df, ref_desc = compute_expiry_gaps(
            df_asof, exp, as_of_date, window, break_even_pct, cutoff_days
        )
        if gaps_df.empty:
            continue

        # Get rows as of the chosen date
        today_rows = gaps_df[gaps_df["date"] == as_of_date].copy()
        if today_rows.empty:
            continue

        for _, r in today_rows.iterrows():
            if r["signal_code"] == "SIGNAL":
                all_signals.append({
                    "symbol": r["symbol"],
                    "expiry": exp.strftime("%Y-%m-%d"),
                    "excess": r["excess"],
                    "status": r["status"],
                })

        # Reference price on this date
        ref_val = today_rows["reference_price"].iloc[0]

        # Format display table for this expiry
        display_df = today_rows[[
            "symbol",
            "price_per_gram",
            "reference_used",
            "gap_pct",
            "normal_level",
            "excess",
            "days_to_expiry",
            "status",
        ]].copy()

        display_df["break_even"] = break_even_pct
        display_df.rename(columns={
            "symbol": "Contract",
            "price_per_gram": "Price (₹/g)",
            "reference_used": "Reference Used",
            "gap_pct": "Gap vs Ref (%)",
            "normal_level": "Normal Level (%)",
            "excess": "Excess Over Normal (%)",
            "break_even": "Break-Even (assumed %)",
            "days_to_expiry": "Days to Expiry",
            "status": "Status",
        }, inplace=True)

        tables_per_expiry.append({
            "expiry": exp.strftime("%Y-%m-%d"),
            "reference_desc": ref_desc,
            "reference_price": ref_val,
            "table": display_df,
        })

    # Overall banner
    if all_signals:
        sig = all_signals[0]
        direction = "above" if sig["excess"] > 0 else "below"
        banner = {
            "code": "SIGNAL",
            "message": f"Signal: {sig['symbol']} ({sig['expiry']}), {abs(sig['excess']):.2f}% {direction} its normal level",
            "count": len(all_signals),
        }
    else:
        banner = {
            "code": "NO_SIGNAL",
            "message": "No signal today",
            "count": 0,
        }

    return banner, tables_per_expiry


# -----------------------------------------------------------------------------
# TAB 3: SETTLEMENT CONVERGENCE TABLE
# -----------------------------------------------------------------------------
def compute_settlement_convergence(
    df_clean: pd.DataFrame, df_ddr: pd.DataFrame
) -> pd.DataFrame:
    """Compare last traded close per gram with DDR per gram for products in ddr.csv."""
    if df_clean.empty or df_ddr.empty:
        return pd.DataFrame()

    results = []
    for _, row in df_ddr.iterrows():
        sym = row["symbol"]
        exp = row["expiry_date"]
        ddr_pg = row["ddr_per_gram"]

        # Find last traded day for this contract
        contract_data = df_clean[
            (df_clean["symbol"] == sym) & (df_clean["expiry_date"] == exp)
        ]
        if contract_data.empty:
            continue

        last_row = contract_data.sort_values("date").iloc[-1]
        last_trade_date = last_row["date"]
        last_close_pg = last_row["close_per_gram"]

        diff_rs = last_close_pg - ddr_pg
        diff_pct = (last_close_pg / ddr_pg - 1.0) * 100.0 if ddr_pg > 0 else np.nan

        results.append({
            "symbol": sym,
            "expiry_date": exp.strftime("%Y-%m-%d"),
            "last_trade_date": last_trade_date.strftime("%Y-%m-%d"),
            "last_close_per_gram": last_close_pg,
            "ddr_per_gram": ddr_pg,
            "diff_rupees_per_gram": diff_rs,
            "diff_pct": diff_pct,
        })

    return pd.DataFrame(results)


# -----------------------------------------------------------------------------
# TAB 5: CALENDAR & LIQUIDITY
# -----------------------------------------------------------------------------
def compute_calendar_timeline(
    df_clean: pd.DataFrame, df_trades: Optional[pd.DataFrame] = None
) -> Tuple[List[Dict[str, Any]], pd.DataFrame]:
    """
    Build lifecycle timeline and volume table for all four contracts.
    Shades tender period (last tender_days) and no new entries (last 10 days).
    """
    if df_clean.empty:
        return [], pd.DataFrame()

    contracts_info = []
    vol_records = []

    # Group by (symbol, expiry_date)
    grouped = df_clean.groupby(["symbol", "expiry_date"])
    for (sym, exp), g in grouped:
        first_date = g["date"].min()
        last_date = g["date"].max()
        expiry_date = exp
        specs = CONTRACT_SPECS.get(sym, {})
        tender_days = specs.get("tender_days", 3)
        grams_per_lot = specs.get("grams_per_lot", 1)

        # Average volume in physical grams
        avg_lots = g["volume_lots"].mean()
        avg_grams = avg_lots * grams_per_lot

        vol_records.append({
            "Contract": sym,
            "Expiry": exp.strftime("%Y-%m-%d"),
            "Lot Size (g)": grams_per_lot,
            "Avg Daily Lots": avg_lots,
            "Avg Daily Volume (g)": avg_grams,
            "Avg Daily Volume (kg)": avg_grams / 1000.0,
            "Trading Days": len(g),
        })

        # Tender period: last tender_days trading days up to and including expiry date
        if last_date >= expiry_date:
            # Finished contract: use actual last tender_days trading dates from data
            sorted_dates = sorted(g["date"].unique())
            tender_start = sorted_dates[-tender_days] if len(sorted_dates) >= tender_days else sorted_dates[0]
            tender_is_estimated = False
        else:
            # Running contract: count back tender_days over weekdays only (Mon-Fri) from expiry
            b_dates = pd.bdate_range(end=expiry_date, periods=tender_days)
            tender_start = b_dates[0]
            tender_is_estimated = True

        # No-new-entries zone: last 10 calendar days before expiry
        no_entry_start = expiry_date - pd.Timedelta(days=10)

        contracts_info.append({
            "symbol": sym,
            "expiry": exp,
            "expiry_str": exp.strftime("%Y-%m-%d"),
            "first_trade": first_date,
            "last_trade": last_date,
            "no_entry_start": no_entry_start,
            "tender_start": tender_start,
            "tender_is_estimated": tender_is_estimated,
        })

    # Liquidty table
    df_vol = pd.DataFrame(vol_records).sort_values("Avg Daily Volume (g)", ascending=False).reset_index(drop=True)
    if not df_vol.empty:
        min_vol = df_vol["Avg Daily Volume (g)"].min()
        df_vol["Liquidity Flag"] = df_vol["Avg Daily Volume (g)"].apply(
            lambda v: "⚠️ Thin" if v == min_vol else "Normal"
        )

    return contracts_info, df_vol
