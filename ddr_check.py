"""
ddr_check.py - Last Traded Price vs Final Settlement Price (Due Date Rate / DDR)

Step 5 of Commodity Derivatives Intelligence project.
Analyzes expiry convergence against MCX official Due Date Rates (DDR) for 999 gold contracts.

Usage:
    python ddr_check.py out results_ddr
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    # -------------------------------------------------------------
    # Step 0: Setup paths and create DDR data file
    # -------------------------------------------------------------
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "out"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "results_ddr"

    os.makedirs(output_folder, exist_ok=True)

    clean_file_path = Path(input_folder) / "gold_futures_clean.csv"
    if not clean_file_path.exists():
        print(f"Error: clean data file not found at {clean_file_path}")
        return

    # Official MCX Due Date Rates (DDR) for 999 contracts
    ddr_records = [
        {"symbol": "GOLDTEN", "expiry_date": "2026-08-31", "ddr_price": 155738.00, "unit_grams": 10},
        {"symbol": "GOLDGUINEA", "expiry_date": "2026-08-31", "ddr_price": 124590.00, "unit_grams": 8},
        {"symbol": "GOLDPETAL", "expiry_date": "2026-08-31", "ddr_price": 15534.00, "unit_grams": 1},
        {"symbol": "GOLDTEN", "expiry_date": "2026-09-30", "ddr_price": 148206.00, "unit_grams": 10},
        {"symbol": "GOLDGUINEA", "expiry_date": "2026-09-30", "ddr_price": 118565.00, "unit_grams": 8},
        {"symbol": "GOLDPETAL", "expiry_date": "2026-09-30", "ddr_price": 14819.00, "unit_grams": 1},
    ]

    ddr_df = pd.DataFrame(ddr_records)
    ddr_csv_path = Path(output_folder) / "ddr.csv"
    ddr_df.to_csv(ddr_csv_path, index=False)
    print(f"Created DDR reference file at: {ddr_csv_path}")

    # Compute ddr_per_gram
    ddr_df["ddr_per_gram"] = ddr_df["ddr_price"] / ddr_df["unit_grams"]

    # Load cleaned data
    df = pd.read_csv(clean_file_path)

    # Filter to only the target 6 contracts and traded days
    target_expiries = ["2026-08-31", "2026-09-30"]
    target_symbols = ["GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]

    df_filtered = df[
        (df["expiry_date"].isin(target_expiries)) &
        (df["symbol"].isin(target_symbols)) &
        (df["traded"] == True)
    ].copy()

    # -------------------------------------------------------------
    # PART 1: Convergence Table
    # -------------------------------------------------------------
    convergence_rows = []

    for _, r in ddr_df.iterrows():
        sym = r["symbol"]
        exp = r["expiry_date"]
        unit = r["unit_grams"]
        ddr_pg = r["ddr_per_gram"]

        contract_history = df_filtered[
            (df_filtered["symbol"] == sym) &
            (df_filtered["expiry_date"] == exp)
        ].sort_values("date")

        if contract_history.empty:
            print(f"Warning: No traded data found for {sym} {exp}")
            continue

        last_row = contract_history.iloc[-1]
        last_date = last_row["date"]
        last_close_pg = last_row["close_per_gram"]

        # Check if last traded date matches expiry date
        if last_date != exp:
            print(f"WARNING: {sym} ({exp}) last traded date {last_date} does NOT match expiry {exp}!")

        diff_rupees_pg = last_close_pg - ddr_pg
        diff_pct = (diff_rupees_pg / ddr_pg) * 100.0

        convergence_rows.append({
            "symbol": sym,
            "expiry_date": exp,
            "last_trade_date": last_date,
            "last_close_per_gram": round(last_close_pg, 3),
            "ddr_per_gram": round(ddr_pg, 3),
            "diff_rupees_per_gram": round(diff_rupees_pg, 3),
            "diff_pct": round(diff_pct, 4)
        })

    conv_df = pd.DataFrame(convergence_rows)
    conv_csv_path = Path(output_folder) / "ddr_convergence.csv"
    conv_df.to_csv(conv_csv_path, index=False)

    print("\n" + "=" * 80)
    print("PART 1: CONVERGENCE TABLE (LAST TRADED CLOSE vs DDR)")
    print("=" * 80)
    print(conv_df.to_string(index=False))

    # -------------------------------------------------------------
    # PART 2: The Gap Before vs At Settlement
    # -------------------------------------------------------------
    gap_comp_rows = []

    for exp in target_expiries:
        exp_conv = conv_df[conv_df["expiry_date"] == exp].set_index("symbol")

        # Gaps at last traded close
        ten_close = exp_conv.loc["GOLDTEN", "last_close_per_gram"]
        guinea_close = exp_conv.loc["GOLDGUINEA", "last_close_per_gram"]
        petal_close = exp_conv.loc["GOLDPETAL", "last_close_per_gram"]

        g_vs_t_close = (guinea_close / ten_close - 1.0) * 100.0
        p_vs_t_close = (petal_close / ten_close - 1.0) * 100.0

        # Gaps at DDR
        ten_ddr = exp_conv.loc["GOLDTEN", "ddr_per_gram"]
        guinea_ddr = exp_conv.loc["GOLDGUINEA", "ddr_per_gram"]
        petal_ddr = exp_conv.loc["GOLDPETAL", "ddr_per_gram"]

        g_vs_t_ddr = (guinea_ddr / ten_ddr - 1.0) * 100.0
        p_vs_t_ddr = (petal_ddr / ten_ddr - 1.0) * 100.0

        gap_comp_rows.append({
            "expiry_date": exp,
            "basis": "Last Traded Close",
            "guinea_vs_ten_pct": round(g_vs_t_close, 4),
            "petal_vs_ten_pct": round(p_vs_t_close, 4)
        })
        gap_comp_rows.append({
            "expiry_date": exp,
            "basis": "Final Settlement (DDR)",
            "guinea_vs_ten_pct": round(g_vs_t_ddr, 4),
            "petal_vs_ten_pct": round(p_vs_t_ddr, 4)
        })

    gap_comp_df = pd.DataFrame(gap_comp_rows)
    gap_comp_csv_path = Path(output_folder) / "gap_close_vs_ddr.csv"
    gap_comp_df.to_csv(gap_comp_csv_path, index=False)

    print("\n" + "=" * 80)
    print("PART 2: GAPS BEFORE vs AT SETTLEMENT")
    print("=" * 80)
    print(gap_comp_df.to_string(index=False))

    # -------------------------------------------------------------
    # PART 3: What-If Pair Trade Held to Settlement
    # -------------------------------------------------------------
    ddr_dict = ddr_df.set_index(["symbol", "expiry_date"])["ddr_price"].to_dict()

    pair_configs = [
        {
            "pair": "PETAL",
            "other_sym": "GOLDPETAL",
            "ten_lots": 1,
            "other_lots": 10,
            "total_grams": 10
        },
        {
            "pair": "GUINEA",
            "other_sym": "GOLDGUINEA",
            "ten_lots": 4,
            "other_lots": 5,
            "total_grams": 40
        }
    ]

    whatif_rows = []

    for exp in target_expiries:
        exp_data = df_filtered[df_filtered["expiry_date"] == exp]
        piv = exp_data.pivot(index="date", columns="symbol", values=["close", "close_per_gram"]).dropna()
        piv.columns = ["_".join(c) for c in piv.columns]
        piv = piv.reset_index().sort_values("date").reset_index(drop=True)

        last_idx = len(piv) - 1

        for p_cfg in pair_configs:
            pair_name = p_cfg["pair"]
            other_sym = p_cfg["other_sym"]
            ten_lots = p_cfg["ten_lots"]
            other_lots = p_cfg["other_lots"]
            tot_grams = p_cfg["total_grams"]

            ten_ddr_price = ddr_dict[("GOLDTEN", exp)]
            other_ddr_price = ddr_dict[(other_sym, exp)]

            for N in [1, 3, 5, 10]:
                entry_idx = last_idx - N
                if entry_idx < 0:
                    print(f"Warning: Not enough trading days for N={N} in {exp}")
                    continue

                entry_row = piv.iloc[entry_idx]
                entry_date = entry_row["date"]

                ten_entry_close = entry_row["close_GOLDTEN"]
                other_entry_close = entry_row[f"close_{other_sym}"]

                ten_entry_pg = entry_row["close_per_gram_GOLDTEN"]
                other_entry_pg = entry_row[f"close_per_gram_{other_sym}"]

                gap_entry_pct = (other_entry_pg / ten_entry_pg - 1.0) * 100.0

                # Gross P&L in rupees:
                # long TEN leg = (ten_ddr_price - ten_entry_close) * number_of_TEN_lots
                # short leg    = -(other_ddr_price - other_entry_close) * number_of_other_lots
                long_ten_leg = (ten_ddr_price - ten_entry_close) * ten_lots
                short_other_leg = -(other_ddr_price - other_entry_close) * other_lots
                gross_pnl_rupees = long_ten_leg + short_other_leg

                # P&L per gram of gold and as percent of long-leg entry value
                pnl_per_gram = gross_pnl_rupees / tot_grams
                long_entry_val = ten_entry_close * ten_lots
                pnl_pct = (gross_pnl_rupees / long_entry_val) * 100.0

                whatif_rows.append({
                    "pair": pair_name,
                    "expiry_date": exp,
                    "days_before_last_day": N,
                    "entry_date": entry_date,
                    "gap_at_entry_pct": round(gap_entry_pct, 4),
                    "gross_pnl_rupees": round(gross_pnl_rupees, 2),
                    "pnl_per_gram": round(pnl_per_gram, 2),
                    "pnl_pct": round(pnl_pct, 4)
                })

    whatif_df = pd.DataFrame(whatif_rows)
    whatif_csv_path = Path(output_folder) / "pair_whatif.csv"
    whatif_df.to_csv(whatif_csv_path, index=False)

    print("\n" + "=" * 80)
    print("PART 3: WHAT-IF PAIR TRADES HELD TO SETTLEMENT (GROSS P&L)")
    print("=" * 80)
    print(whatif_df.to_string(index=False))
    print("\nCosts are NOT included. Holding to expiry means compulsory physical delivery, extra delivery margin, and making charges that are not modelled here.")
    print("Earlier assumed cost scenario, for reference only: ~0.22% of position value round-trip.")

    # -------------------------------------------------------------
    # PART 4: Chart (gap_to_settlement.png)
    # -------------------------------------------------------------
    fig, axes = plt.subplots(len(target_expiries), 1, figsize=(12, 5.5 * len(target_expiries)), squeeze=False)

    for idx, exp in enumerate(target_expiries):
        ax = axes[idx, 0]
        exp_data = df_filtered[df_filtered["expiry_date"] == exp]
        piv_chart = exp_data.pivot(index="date", columns="symbol", values="close_per_gram").dropna()

        # Last 15 trading days
        last15 = piv_chart.tail(15).copy()
        last15["guinea_vs_ten"] = (last15["GOLDGUINEA"] / last15["GOLDTEN"] - 1.0) * 100.0
        last15["petal_vs_ten"] = (last15["GOLDPETAL"] / last15["GOLDTEN"] - 1.0) * 100.0

        dates_str = list(last15.index)

        # Plot close lines
        ax.plot(dates_str, last15["guinea_vs_ten"], marker="o", color="#1f77b4", lw=2, label="GOLDGUINEA vs GOLDTEN (Close)")
        ax.plot(dates_str, last15["petal_vs_ten"], marker="s", color="#ff7f0e", lw=2, linestyle="--", label="GOLDPETAL vs GOLDTEN (Close)")

        # Fetch DDR gaps for this expiry
        exp_conv = conv_df[conv_df["expiry_date"] == exp].set_index("symbol")
        ten_ddr = exp_conv.loc["GOLDTEN", "ddr_per_gram"]
        g_ddr_gap = (exp_conv.loc["GOLDGUINEA", "ddr_per_gram"] / ten_ddr - 1.0) * 100.0
        p_ddr_gap = (exp_conv.loc["GOLDPETAL", "ddr_per_gram"] / ten_ddr - 1.0) * 100.0

        # Plot DDR points at the final date
        last_date_str = dates_str[-1]
        ax.scatter([last_date_str], [g_ddr_gap], color="#1f77b4", s=180, marker="*", edgecolor="black", zorder=5, label=f"GUINEA DDR ({g_ddr_gap:+.3f}%)")
        ax.scatter([last_date_str], [p_ddr_gap], color="#ff7f0e", s=180, marker="D", edgecolor="black", zorder=5, label=f"PETAL DDR ({p_ddr_gap:+.3f}%)")

        # Dashed drop lines from last close to DDR
        last_g_close = last15["guinea_vs_ten"].iloc[-1]
        last_p_close = last15["petal_vs_ten"].iloc[-1]
        ax.plot([last_date_str, last_date_str], [last_g_close, g_ddr_gap], color="#1f77b4", linestyle=":", lw=2)
        ax.plot([last_date_str, last_date_str], [last_p_close, p_ddr_gap], color="#ff7f0e", linestyle=":", lw=2)

        ax.axhline(0.0, color="gray", linestyle="-", lw=1.0, alpha=0.7)
        ax.set_title(f"Price Spread Gaps Leading to Final DDR Settlement (Expiry: {exp})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Trading Date", fontsize=10)
        ax.set_ylabel("Gap vs GOLDTEN (%)", fontsize=10)
        ax.tick_params(axis="x", rotation=40)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = Path(output_folder) / "gap_to_settlement.png"
    plt.savefig(chart_path, dpi=200)
    plt.close()
    print(f"\nSaved convergence chart to: {chart_path}")

    # -------------------------------------------------------------
    # PART 5: Audit (Print ONE what-if trade in full detail)
    # -------------------------------------------------------------
    sample = whatif_df[(whatif_df["pair"] == "PETAL") & (whatif_df["days_before_last_day"] == 5)].iloc[0]
    sample_exp = sample["expiry_date"]
    sample_entry_date = sample["entry_date"]

    # Re-fetch exact prices for sample trade audit
    sample_piv = df_filtered[df_filtered["expiry_date"] == sample_exp].pivot(index="date", columns="symbol", values="close")
    ten_entry_price = sample_piv.loc[sample_entry_date, "GOLDTEN"]
    petal_entry_price = sample_piv.loc[sample_entry_date, "GOLDPETAL"]
    ten_ddr_audit = ddr_dict[("GOLDTEN", sample_exp)]
    petal_ddr_audit = ddr_dict[("GOLDPETAL", sample_exp)]

    ten_leg_audit = (ten_ddr_audit - ten_entry_price) * 1.0
    petal_leg_audit = -(petal_ddr_audit - petal_entry_price) * 10.0
    tot_pnl_audit = ten_leg_audit + petal_leg_audit

    print("\n" + "=" * 80)
    print("PART 5: AUDIT OF ONE WHAT-IF TRADE")
    print("=" * 80)
    print(f"Pair: PETAL | Expiry: {sample_exp} | Entry: {sample_entry_date} (5 trading days before last day)")
    print(f"Leg 1 (Long 1 lot GOLDTEN, 10g):")
    print(f"  Entry Close = Rs {ten_entry_price:,.2f} | Exit DDR = Rs {ten_ddr_audit:,.2f}")
    print(f"  Long Leg P&L = ({ten_ddr_audit:,.2f} - {ten_entry_price:,.2f}) x 1 = Rs {ten_leg_audit:+,.2f}")
    print(f"Leg 2 (Short 10 lots GOLDPETAL, 10g):")
    print(f"  Entry Close = Rs {petal_entry_price:,.2f}/g | Exit DDR = Rs {petal_ddr_audit:,.2f}/g")
    print(f"  Short Leg P&L = -({petal_ddr_audit:,.2f} - {petal_entry_price:,.2f}) x 10 = Rs {petal_leg_audit:+,.2f}")
    print(f"Total Gross P&L: Rs {tot_pnl_audit:+,.2f}")
    print(f"Hedged Gold Weight: 10 grams | P&L per Gram: Rs {tot_pnl_audit / 10.0:+,.2f}/g")
    print(f"Long-Leg Value at Entry: Rs {ten_entry_price:,.2f} | Return on Long Leg: {(tot_pnl_audit / ten_entry_price) * 100.0:+.4f}%")

    # -------------------------------------------------------------
    # SUMMARY (Plain-English, at most 8 lines, built from results)
    # -------------------------------------------------------------
    ten_conv = conv_df[conv_df["symbol"] == "GOLDTEN"]
    guinea_conv = conv_df[conv_df["symbol"] == "GOLDGUINEA"]
    petal_conv = conv_df[conv_df["symbol"] == "GOLDPETAL"]

    max_ten_diff = ten_conv["diff_pct"].abs().max()
    min_g_diff_rs = guinea_conv["diff_rupees_per_gram"].min()
    max_g_diff_rs = guinea_conv["diff_rupees_per_gram"].max()
    min_p_diff_rs = petal_conv["diff_rupees_per_gram"].min()
    max_p_diff_rs = petal_conv["diff_rupees_per_gram"].max()

    max_g_close = gap_comp_df[gap_comp_df["basis"] == "Last Traded Close"]["guinea_vs_ten_pct"].max()
    max_g_ddr = gap_comp_df[gap_comp_df["basis"] == "Final Settlement (DDR)"]["guinea_vs_ten_pct"].abs().max()
    max_p_close = gap_comp_df[gap_comp_df["basis"] == "Last Traded Close"]["petal_vs_ten_pct"].max()
    max_p_ddr = gap_comp_df[gap_comp_df["basis"] == "Final Settlement (DDR)"]["petal_vs_ten_pct"].abs().max()

    min_pnl_rs = whatif_df["gross_pnl_rupees"].min()
    max_pnl_rs = whatif_df["gross_pnl_rupees"].max()
    min_pnl_pct = whatif_df["pnl_pct"].min()
    max_pnl_pct = whatif_df["pnl_pct"].max()

    print("\n" + "=" * 80)
    print("PLAIN-ENGLISH SUMMARY")
    print("=" * 80)
    print(f"1. On the final day, GOLDTEN closed within {max_ten_diff:.2f}% of DDR, while GOLDGUINEA (+{min_g_diff_rs:.1f} to +{max_g_diff_rs:.1f} Rs/g) and GOLDPETAL (+{min_p_diff_rs:.1f} to +{max_p_diff_rs:.1f} Rs/g) closed significantly above DDR.")
    print(f"2. Gaps collapse at settlement: GUINEA gap shrank from up to {max_g_close:.2f}% at close down to {max_g_ddr:.4f}% at DDR, and PETAL gap shrank from {max_p_close:.2f}% to {max_p_ddr:.2f}%.")
    print(f"3. What-if trades held to DDR yielded gross P&L between Rs {min_pnl_rs:+.0f} ({min_pnl_pct:+.2f}%) and Rs {max_pnl_rs:+.0f} ({max_pnl_pct:+.2f}%) across 16 scenarios.")
    print(f"4. CAUTION: Findings are based on only {len(target_expiries)} expiries ({', '.join(target_expiries)}).")
    print("5. CAUTION: Trading and physical delivery costs (delivery margins, vaulting, GST) are NOT subtracted here.")
    print("6. CAUTION: The DDR is a polled spot benchmark rate, not an executable live screen quote.")
    print("7. CAUTION: Holding MCX gold contracts into final expiry triggers compulsory physical delivery obligations.")
    print("=" * 80)


if __name__ == "__main__":
    main()
