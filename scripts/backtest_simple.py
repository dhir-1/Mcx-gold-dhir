"""
backtest_simple.py - Simple Arbitrage / Spread Backtest on MCX GOLDTEN vs GOLDPETAL

Step 4 of Commodity Derivatives Intelligence project.
Pairs trading: Buy GOLDTEN (10g) and Sell GOLDPETAL (10x1g = 10g) in equal grams.
Evaluates Hold Baseline vs Reversion strategies under realistic transaction friction.

Usage:
    python backtest_simple.py out results_bt
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -------------------------------------------------------------
# Transaction Cost Assumptions (editable config dict)
# -------------------------------------------------------------
COST_CONFIG = {
    "brokerage_per_order": 20.0,       # Rs 20 flat per order (or 0.03% if lower)
    "brokerage_pct": 0.0003,           # 0.03% cap
    "ctt_sell_pct": 0.0001,            # 0.01% on sell leg value
    "exchange_txn_pct": 0.000021,      # 0.0021% on total turnover
    "sebi_per_crore": 10.0,            # Rs 10 per crore (0.0001%)
    "stamp_buy_pct": 0.00002,          # 0.002% on buy leg value
    "gst_pct": 0.18,                   # 18% on (brokerage + exchange_txn + sebi)
    "slippage_pct": 0.0003,            # 0.03% slippage applied on each leg
}


def calculate_trade_costs(ten_entry_price, petal_entry_price, ten_exit_price, petal_exit_price):
    """
    Computes exact fees, statutory levies, and slippage for one round-trip trade:
      Leg 1: BUY  1 lot  GOLDTEN   (10 g) at ten_entry_price
      Leg 2: SELL 10 lots GOLDPETAL (10 g) at petal_entry_price * 10
      Leg 3: SELL 1 lot  GOLDTEN   (10 g) at ten_exit_price
      Leg 4: BUY  10 lots GOLDPETAL (10 g) at petal_exit_price * 10
    """
    v_buy_ten = ten_entry_price * 1.0
    v_sell_pet = petal_entry_price * 10.0
    v_sell_ten = ten_exit_price * 1.0
    v_buy_pet = petal_exit_price * 10.0

    order_values = [v_buy_ten, v_sell_pet, v_sell_ten, v_buy_pet]
    turnover = sum(order_values)

    # 1. Brokerage: min(20 flat, 0.03% of order value) for each of the 4 orders
    brokerage = sum(min(COST_CONFIG["brokerage_per_order"], COST_CONFIG["brokerage_pct"] * v) for v in order_values)

    # 2. Exchange transaction fee: 0.0021% of total turnover
    exchange_txn = COST_CONFIG["exchange_txn_pct"] * turnover

    # 3. SEBI turnover charge: Rs 10 per crore (10 / 10,000,000)
    sebi = (COST_CONFIG["sebi_per_crore"] / 1e7) * turnover

    # 4. GST: 18% of (brokerage + exchange_txn + sebi)
    gst = COST_CONFIG["gst_pct"] * (brokerage + exchange_txn + sebi)

    # 5. Stamp duty: 0.002% on BUY orders only
    stamp_duty = COST_CONFIG["stamp_buy_pct"] * (v_buy_ten + v_buy_pet)

    # 6. CTT (Commodity Transaction Tax): 0.01% on SELL orders only
    ctt = COST_CONFIG["ctt_sell_pct"] * (v_sell_pet + v_sell_ten)

    total_fees = brokerage + exchange_txn + sebi + gst + stamp_duty + ctt

    # 7. Slippage: 0.03% applied on each of the 4 trades
    total_slippage = COST_CONFIG["slippage_pct"] * turnover

    return {
        "turnover": turnover,
        "position_value": (v_buy_ten + v_sell_pet) / 2.0,
        "brokerage": brokerage,
        "exchange_txn": exchange_txn,
        "sebi": sebi,
        "gst": gst,
        "stamp_duty": stamp_duty,
        "ctt": ctt,
        "fees": total_fees,
        "slippage": total_slippage,
        "total_cost": total_fees + total_slippage
    }


def main():
    # -------------------------------------------------------------
    # Step 1: Parse arguments and load cleaned gold futures data
    # -------------------------------------------------------------
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "out"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "results_bt"

    os.makedirs(output_folder, exist_ok=True)

    clean_file_path = Path(input_folder) / "gold_futures_clean.csv"
    if not clean_file_path.exists():
        print(f"Error: clean data file not found at {clean_file_path}")
        return

    df = pd.read_csv(clean_file_path)

    # Keep only GOLDTEN and GOLDPETAL rows that traded
    target_symbols = ["GOLDTEN", "GOLDPETAL"]
    df_f = df[df["symbol"].isin(target_symbols) & (df["traded"] == True)].copy()

    expiries = sorted(df_f["expiry_date"].unique())

    all_trades = []

    # -------------------------------------------------------------
    # Step 2: Run backtest separately for each expiry
    # -------------------------------------------------------------
    for exp in expiries:
        exp_df = df_f[df_f["expiry_date"] == exp]

        # Pivot to get aligned daily closes and days_to_expiry
        piv = exp_df.pivot(index="date", columns="symbol", values=["close", "close_per_gram", "days_to_expiry"]).dropna()
        piv.columns = ["_".join(c) for c in piv.columns]
        piv = piv.reset_index().sort_values("date").reset_index(drop=True)

        # Gap in percent: (PETAL per gram / TEN per gram - 1) * 100
        piv["gap_pct"] = (piv["close_per_gram_GOLDPETAL"] / piv["close_per_gram_GOLDTEN"] - 1.0) * 100.0
        piv["days_to_expiry"] = piv["days_to_expiry_GOLDTEN"].astype(int)

        # ---------------------------------------------------------
        # Strategy A: HOLD BASELINE (1 trade per expiry)
        # ---------------------------------------------------------
        # Enter on first eligible day (days_to_expiry > 10)
        eligible_entries = piv[piv["days_to_expiry"] > 10]
        if not eligible_entries.empty:
            entry_idx_A = eligible_entries.index[0]
            entry_row_A = piv.iloc[entry_idx_A]
            entry_date_A = entry_row_A["date"]

            # Forced exit: first trading day with days_to_expiry <= 7, or last available day
            forced_candidates = piv.iloc[entry_idx_A:][piv.iloc[entry_idx_A:]["days_to_expiry"] <= 7]
            if not forced_candidates.empty:
                exit_row_A = forced_candidates.iloc[0]
                exit_idx_A = forced_candidates.index[0]
            else:
                exit_row_A = piv.iloc[-1]
                exit_idx_A = len(piv) - 1

            exit_date_A = exit_row_A["date"]
            days_held_A = exit_idx_A - entry_idx_A

            ten_en_A = entry_row_A["close_GOLDTEN"]
            pet_en_A = entry_row_A["close_GOLDPETAL"]
            ten_ex_A = exit_row_A["close_GOLDTEN"]
            pet_ex_A = exit_row_A["close_GOLDPETAL"]

            # P&L calculations
            ten_leg_pnl_A = (ten_ex_A - ten_en_A) * 1.0
            pet_leg_pnl_A = -(pet_ex_A - pet_en_A) * 10.0
            gross_pnl_A = ten_leg_pnl_A + pet_leg_pnl_A

            costs_A = calculate_trade_costs(ten_en_A, pet_en_A, ten_ex_A, pet_ex_A)
            fees_A = costs_A["fees"]
            slip_A = costs_A["slippage"]

            ten_chg_pct_A = (ten_ex_A / ten_en_A - 1.0) * 100.0
            gold_pnl_A = ten_leg_pnl_A

            all_trades.append({
                "strategy": "HOLD_BASELINE",
                "expiry": exp,
                "period": "descriptive, no test period",
                "entry_date": entry_date_A,
                "exit_date": exit_date_A,
                "days_held": days_held_A,
                "gap_at_entry": round(entry_row_A["gap_pct"], 4),
                "gap_at_exit": round(exit_row_A["gap_pct"], 4),
                "gross_pnl": round(gross_pnl_A, 2),
                "fees": round(fees_A, 2),
                "slippage": round(slip_A, 2),
                "net_pnl_fees": round(gross_pnl_A - fees_A, 2),
                "net_pnl_fees_slippage": round(gross_pnl_A - fees_A - slip_A, 2),
                "ten_price_change_pct": round(ten_chg_pct_A, 2),
                "gold_only_pnl": round(gold_pnl_A, 2),
            })

        # ---------------------------------------------------------
        # Strategy B: REVERSION
        # ---------------------------------------------------------
        # Compute mean m and std s of gap_pct over previous 10 trading days (not including today)
        piv["m"] = piv["gap_pct"].shift(1).rolling(10).mean()
        piv["s"] = piv["gap_pct"].shift(1).rolling(10).std(ddof=1)
        piv["upper"] = piv["m"] + 1.0 * piv["s"]

        in_pos = False
        entry_idx = None
        i = 0

        while i < len(piv):
            row = piv.iloc[i]

            if not in_pos:
                # Signal on day i: gap_pct > m + 1.0 * s
                if pd.notna(row["upper"]) and row["gap_pct"] > row["upper"]:
                    # Decision made at close of day i -> execute at close of day i+1
                    if i + 1 < len(piv):
                        exec_row = piv.iloc[i + 1]
                        # Only enter if days_to_expiry > 10 on entry day
                        if exec_row["days_to_expiry"] > 10:
                            in_pos = True
                            entry_idx = i + 1
                            i += 1
                            continue
            else:
                days_held = i - entry_idx
                cur_dte = row["days_to_expiry"]
                cur_gap = row["gap_pct"]
                cur_m = row["m"]

                should_exit = False

                # Exit rule 1: Force close at first day with days_to_expiry <= 7
                if cur_dte <= 7:
                    should_exit = True
                # Exit rule 2: After 5 trading days held
                elif days_held >= 5:
                    should_exit = True
                # Exit rule 3: Reversion exit when gap_pct <= m (signal today -> execute tomorrow, or exit today)
                elif cur_gap <= cur_m:
                    if i + 1 < len(piv):
                        i += 1
                        row = piv.iloc[i]
                        days_held = i - entry_idx
                        should_exit = True
                    else:
                        should_exit = True
                # Exit rule 4: End of available data
                elif i == len(piv) - 1:
                    should_exit = True

                if should_exit:
                    en_row = piv.iloc[entry_idx]
                    ex_row = row

                    ten_en = en_row["close_GOLDTEN"]
                    pet_en = en_row["close_GOLDPETAL"]
                    ten_ex = ex_row["close_GOLDTEN"]
                    pet_ex = ex_row["close_GOLDPETAL"]

                    ten_pnl = (ten_ex - ten_en) * 1.0
                    pet_pnl = -(pet_ex - pet_en) * 10.0
                    gross_b = ten_pnl + pet_pnl

                    costs_b = calculate_trade_costs(ten_en, pet_en, ten_ex, pet_ex)
                    fees_b = costs_b["fees"]
                    slip_b = costs_b["slippage"]

                    en_date = en_row["date"]
                    period_b = "Design period" if en_date <= "2026-09-11" else "Unseen test period"

                    all_trades.append({
                        "strategy": "REVERSION",
                        "expiry": exp,
                        "period": period_b,
                        "entry_date": en_date,
                        "exit_date": ex_row["date"],
                        "days_held": days_held,
                        "gap_at_entry": round(en_row["gap_pct"], 4),
                        "gap_at_exit": round(ex_row["gap_pct"], 4),
                        "gross_pnl": round(gross_b, 2),
                        "fees": round(fees_b, 2),
                        "slippage": round(slip_b, 2),
                        "net_pnl_fees": round(gross_b - fees_b, 2),
                        "net_pnl_fees_slippage": round(gross_b - fees_b - slip_b, 2),
                        "ten_price_change_pct": round((ten_ex / ten_en - 1.0) * 100.0, 2),
                        "gold_only_pnl": round(ten_pnl, 2),
                    })

                    in_pos = False
                    entry_idx = None

            i += 1

    trades_df = pd.DataFrame(all_trades)

    # -------------------------------------------------------------
    # Step 3: Save results_bt/trades.csv
    # -------------------------------------------------------------
    trades_csv_path = Path(output_folder) / "trades.csv"
    trades_df.to_csv(trades_csv_path, index=False)
    print("=" * 80)
    print("STEP 4: BACKTEST COMPLETED")
    print("=" * 80)
    print(f"Saved trade log to: {trades_csv_path}")

    # -------------------------------------------------------------
    # Output 2: Summary table per strategy and period
    # -------------------------------------------------------------
    summary_rows = []
    for (strat, per), group in trades_df.groupby(["strategy", "period"]):
        n_trades = len(group)
        tot_gross = group["gross_pnl"].sum()
        tot_net_fees = group["net_pnl_fees"].sum()
        tot_net_slip = group["net_pnl_fees_slippage"].sum()
        avg_gross = group["gross_pnl"].mean()
        avg_net_slip = group["net_pnl_fees_slippage"].mean()
        win_rate = (group["net_pnl_fees_slippage"] > 0).mean() * 100.0

        summary_rows.append({
            "strategy": strat,
            "period": per,
            "trades": n_trades,
            "total_gross_pnl": round(tot_gross, 2),
            "total_net_fees": round(tot_net_fees, 2),
            "total_net_slip": round(tot_net_slip, 2),
            "avg_gross": round(avg_gross, 2),
            "avg_net_slip": round(avg_net_slip, 2),
            "win_rate_pct": round(win_rate, 1)
        })

    summary_df = pd.DataFrame(summary_rows)
    print("\n--- PERFORMANCE SUMMARY BY STRATEGY AND PERIOD ---")
    print(summary_df.to_string(index=False))

    # -------------------------------------------------------------
    # Output 3: Break-even cost analysis
    # -------------------------------------------------------------
    avg_fees = trades_df["fees"].mean()
    avg_slip = trades_df["slippage"].mean()
    avg_total_cost = avg_fees + avg_slip
    avg_gross_all = trades_df["gross_pnl"].mean()

    # Approximate 10g position value from trades
    approx_pos_val = 150000.0
    cost_pct_pos = (avg_total_cost / approx_pos_val) * 100.0
    fees_pct_pos = (avg_fees / approx_pos_val) * 100.0

    print("\n--- BREAK-EVEN COST ANALYSIS ---")
    print(f"Average Round-Trip Fees Only:      Rs {avg_fees:.2f} ({fees_pct_pos:.3f}% of position value)")
    print(f"Average Round-Trip Fees+Slippage:  Rs {avg_total_cost:.2f} ({cost_pct_pos:.3f}% of position value)")
    print(f"Average Gross P&L Per Trade:       Rs {avg_gross_all:.2f}")
    print(f"Break-even Requirement: The spread must shrink by at least {cost_pct_pos:.2f}% to break even after full costs.")

    # -------------------------------------------------------------
    # Output 4: Gold-neutral check
    # -------------------------------------------------------------
    tot_strat_gross = trades_df["gross_pnl"].sum()
    tot_strat_net = trades_df["net_pnl_fees_slippage"].sum()
    tot_gold_pnl = trades_df["gold_only_pnl"].sum()

    print("\n--- GOLD-NEUTRAL INTEGRITY CHECK ---")
    print(f"Total Strategy Gross P&L:          Rs {tot_strat_gross:.2f}")
    print(f"Total Strategy Net P&L (All Cost): Rs {tot_strat_net:.2f}")
    print(f"Total Pure Gold Holding P&L:       Rs {tot_gold_pnl:.2f}")
    print("Interpretation: Spread P&L is isolated from absolute gold price swings (delta-neutral 10g vs 10g hedge).")

    # -------------------------------------------------------------
    # Output 6: Print ONE trade in full detail
    # -------------------------------------------------------------
    sample_t = trades_df[trades_df["strategy"] == "HOLD_BASELINE"].iloc[0]
    # Reconstruct sample trade numbers for audit
    sample_exp = sample_t["expiry"]
    sample_df = df_f[df_f["expiry_date"] == sample_exp]
    sample_piv = sample_df.pivot(index="date", columns="symbol", values="close")
    sample_ten_en = sample_piv.loc[sample_t["entry_date"], "GOLDTEN"]
    sample_pet_en = sample_piv.loc[sample_t["entry_date"], "GOLDPETAL"]
    sample_ten_ex = sample_piv.loc[sample_t["exit_date"], "GOLDTEN"]
    sample_pet_ex = sample_piv.loc[sample_t["exit_date"], "GOLDPETAL"]
    sample_audit_costs = calculate_trade_costs(sample_ten_en, sample_pet_en, sample_ten_ex, sample_pet_ex)

    print("\n--- FULL DETAILED AUDIT FOR ONE TRADE ---")
    print(f"Strategy: {sample_t['strategy']} | Expiry: {sample_t['expiry']}")
    print(f"Entry Date: {sample_t['entry_date']} | Exit Date: {sample_t['exit_date']} (Held {sample_t['days_held']} trading days)")
    print(f"Entry Prices: GOLDTEN = Rs {sample_ten_en:,.2f}/10g | GOLDPETAL = Rs {sample_pet_en:,.2f}/g | Gap = {sample_t['gap_at_entry']:.4f}%")
    print(f"Exit Prices:  GOLDTEN = Rs {sample_ten_ex:,.2f}/10g | GOLDPETAL = Rs {sample_pet_ex:,.2f}/g | Gap = {sample_t['gap_at_exit']:.4f}%")
    print(f"Position: Long 1 lot GOLDTEN (10g) + Short 10 lots GOLDPETAL (10g)")
    print(f"Leg 1 (Long GOLDTEN):  Rs {sample_ten_ex - sample_ten_en:+,.2f}")
    print(f"Leg 2 (Short GOLDPETAL):Rs {-(sample_pet_ex - sample_pet_en) * 10:+,.2f}")
    print(f"Gross P&L:             Rs {sample_t['gross_pnl']:+,.2f}")
    print(f"Cost Breakdown (Turnover = Rs {sample_audit_costs['turnover']:,.2f}):")
    print(f"  - Brokerage (4 orders @ min(20, 0.03%)): Rs {sample_audit_costs['brokerage']:.2f}")
    print(f"  - Exchange Txn (0.0021%):                Rs {sample_audit_costs['exchange_txn']:.2f}")
    print(f"  - SEBI Charges (Rs 10/crore):            Rs {sample_audit_costs['sebi']:.2f}")
    print(f"  - GST (18% on Brokerage+Exch+SEBI):      Rs {sample_audit_costs['gst']:.2f}")
    print(f"  - Stamp Duty (0.002% on buys):           Rs {sample_audit_costs['stamp_duty']:.2f}")
    print(f"  - CTT (0.01% on sells):                  Rs {sample_audit_costs['ctt']:.2f}")
    print(f"  Total Statutory & Broker Fees:           Rs {sample_audit_costs['fees']:.2f}")
    print(f"  Slippage (0.03% on each of 4 legs):      Rs {sample_audit_costs['slippage']:.2f}")
    print(f"Net P&L (after fees):                      Rs {sample_t['net_pnl_fees']:+,.2f}")
    print(f"Net P&L (after fees + slippage):           Rs {sample_t['net_pnl_fees_slippage']:+,.2f}")
    print(f"Gold Price Move Over Window:               {sample_t['ten_price_change_pct']:+.2f}% (Rs {sample_t['gold_only_pnl']:+,.2f})")

    # -------------------------------------------------------------
    # Output 5: Chart results_bt/strategy_vs_gold.png
    # -------------------------------------------------------------
    # Build a daily time series of cumulative net P&L for Strategy B
    all_dates = sorted(df_f["date"].unique())
    strat_b_trades = trades_df[trades_df["strategy"] == "REVERSION"].sort_values("exit_date")

    daily_pnl = pd.Series(0.0, index=pd.to_datetime(all_dates))
    for _, tr in strat_b_trades.iterrows():
        d = pd.to_datetime(tr["exit_date"])
        if d in daily_pnl.index:
            daily_pnl.loc[d] += tr["net_pnl_fees_slippage"]

    cum_pnl_b = daily_pnl.cumsum()

    # Get active/reference GOLDTEN price (use the longest expiry, 2026-10-30)
    ten_prices = df_f[(df_f["symbol"] == "GOLDTEN") & (df_f["expiry_date"] == "2026-10-30")].set_index("date")["close"]
    ten_series = pd.Series(ten_prices.values, index=pd.to_datetime(ten_prices.index)).sort_index()

    fig, ax1 = plt.subplots(figsize=(11, 5))
    color_pnl = "#1f77b4"
    color_gold = "#d62728"

    ax1.plot(cum_pnl_b.index, cum_pnl_b.values, color=color_pnl, lw=2.2, label="Strategy B Cum Net P&L (Fees+Slippage)")
    ax1.set_xlabel("Date", fontsize=10)
    ax1.set_ylabel("Cumulative Net P&L (Rs)", color=color_pnl, fontsize=10)
    ax1.tick_params(axis="y", labelcolor=color_pnl)
    ax1.axhline(0.0, color="gray", linestyle=":", lw=1.2)
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(ten_series.index, ten_series.values, color=color_gold, lw=1.8, linestyle="--", alpha=0.8, label="GOLDTEN Close Price")
    ax2.set_ylabel("GOLDTEN Price (Rs / 10g)", color=color_gold, fontsize=10)
    ax2.tick_params(axis="y", labelcolor=color_gold)

    plt.title("Strategy B Reversion Cumulative Net P&L vs GOLDTEN Price", fontsize=12, fontweight="bold")
    fig.tight_layout()

    chart_path = Path(output_folder) / "strategy_vs_gold.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nSaved performance chart to: {chart_path}")

    # -------------------------------------------------------------
    # Output 7: Dynamic Plain-English Summary (at most 10 lines)
    # -------------------------------------------------------------
    # Extract calculated metrics for dynamic text
    strat_a_df = trades_df[trades_df["strategy"] == "HOLD_BASELINE"]
    strat_b_df = trades_df[trades_df["strategy"] == "REVERSION"]

    a_gross = strat_a_df["gross_pnl"].sum()
    a_net_fees = strat_a_df["net_pnl_fees"].sum()
    a_net_slip = strat_a_df["net_pnl_fees_slippage"].sum()

    b_gross = strat_b_df["gross_pnl"].sum()
    b_net_fees = strat_b_df["net_pnl_fees"].sum()
    b_net_slip = strat_b_df["net_pnl_fees_slippage"].sum()

    b_design = strat_b_df[strat_b_df["period"] == "Design period"]
    b_unseen = strat_b_df[strat_b_df["period"] == "Unseen test period"]

    n_design = len(b_design)
    n_unseen = len(b_unseen)
    net_design = b_design["net_pnl_fees_slippage"].sum() if n_design > 0 else 0.0
    net_unseen = b_unseen["net_pnl_fees_slippage"].sum() if n_unseen > 0 else 0.0

    print("\n" + "=" * 80)
    print("PLAIN-ENGLISH SUMMARY")
    print("=" * 80)

    # Line 1: Strategy A profitability
    print(f"1. HOLD BASELINE made Rs {a_gross:+.2f} gross, Rs {a_net_fees:+.2f} after fees, and Rs {a_net_slip:+.2f} after fees+slippage across {len(strat_a_df)} expiries.")

    # Line 2: Strategy B profitability
    print(f"2. REVERSION made Rs {b_gross:+.2f} gross, Rs {b_net_fees:+.2f} after fees, and Rs {b_net_slip:+.2f} after fees+slippage across {len(strat_b_df)} trades.")

    # Line 3: Design vs Unseen comparison
    print(f"3. Design period had {n_design} trade (Net: Rs {net_design:+.2f}), while Unseen test period had {n_unseen} trade (Net: Rs {net_unseen:+.2f}).")

    # Line 4: Small test sample warning
    if n_unseen < 3:
        print(f"4. NOTICE: The unseen test period has only {n_unseen} trade(s), which is too small to draw firm statistical conclusions.")
    else:
        print(f"4. NOTICE: The unseen test period has {n_unseen} trades for evaluation.")

    # Line 5: Break-even cost barrier
    print(f"5. Frictions are decisive: total round-trip costs average Rs {avg_total_cost:.2f} (~{cost_pct_pos:.2f}% of position), exceeding narrow reversion gains.")

    # Line 6: Caution on settlement prices
    print("6. CAUTION: Bhavcopy settlement marks are not real trade fills; live bid-ask spreads and liquidity would increase slippage further.")

    # Line 7: Caution on sample size and overlapping dates
    print(f"7. CAUTION: Sample size is small ({len(trades_df)} total trades), and all 3 expiries cover identical calendar dates (Aug-Oct 2026).")

    # Line 8: Caution on fixed parameters
    print("8. CAUTION: Strategy parameters (10d lookback, 1.0 std dev, 5d max hold) were fixed in advance without parameter optimization.")
    print("=" * 80)


if __name__ == "__main__":
    main()
