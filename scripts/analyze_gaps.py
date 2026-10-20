"""
analyze_gaps.py - Measure Price Gaps Between MCX 999 Gold Futures Contracts

Step 2 of Commodity Derivatives Intelligence project.
Analyzes normalized price-per-gram spreads across GOLDTEN, GOLDGUINEA, and GOLDPETAL.

Usage:
    python analyze_gaps.py out results
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
    # Parse CLI arguments and set input / output paths
    # -------------------------------------------------------------
    input_folder = sys.argv[1] if len(sys.argv) > 1 else "out"
    output_folder = sys.argv[2] if len(sys.argv) > 2 else "results"

    # Create the results directory if it does not already exist
    os.makedirs(output_folder, exist_ok=True)

    clean_file_path = Path(input_folder) / "gold_futures_clean.csv"
    if not clean_file_path.exists():
        print(f"Error: Could not find clean data file at '{clean_file_path}'")
        return

    # Load cleaned gold futures data
    df = pd.read_csv(clean_file_path)

    # -------------------------------------------------------------
    # Filter scope: Keep only 999 purity contracts that traded
    # -------------------------------------------------------------
    target_symbols = ["GOLDTEN", "GOLDGUINEA", "GOLDPETAL"]
    df_filtered = df[df["symbol"].isin(target_symbols) & (df["traded"] == True)].copy()

    # Define lot size in grams to calculate physical volume in grams
    grams_per_lot = {
        "GOLDTEN": 10,
        "GOLDGUINEA": 8,
        "GOLDPETAL": 1
    }
    df_filtered["grams_per_lot"] = df_filtered["symbol"].map(grams_per_lot)
    df_filtered["volume_grams"] = df_filtered["volume_lots"] * df_filtered["grams_per_lot"]

    # Identify all unique expiries present in the filtered data
    unique_expiries = sorted(df_filtered["expiry_date"].unique())

    all_daily_gaps = []
    all_gap_stats = []
    all_bucket_stats = []
    valid_expiries = []

    print("=" * 75)
    print("STEP 2: MEASURING PRICE GAPS BETWEEN 999 GOLD CONTRACTS")
    print("=" * 75)

    # -------------------------------------------------------------
    # Step 1 & 2: Build daily table & compute percentage gaps per expiry
    # -------------------------------------------------------------
    for exp in unique_expiries:
        exp_df = df_filtered[df_filtered["expiry_date"] == exp]

        # Pivot close_per_gram with date as index and symbols as columns
        piv_price = exp_df.pivot(index="date", columns="symbol", values="close_per_gram")

        # Keep only dates where ALL THREE contracts have traded prices
        piv_price = piv_price.dropna(subset=target_symbols).copy()

        if piv_price.empty:
            print(f"Skipping expiry {exp}: Not all 3 contracts traded on the same dates.")
            continue

        valid_expiries.append(exp)

        # Pivot volume in grams to carry over for liquidity checks
        piv_vol_grams = exp_df.pivot(index="date", columns="symbol", values="volume_grams")[target_symbols]
        piv_vol_grams.columns = [f"{s}_vol_grams" for s in piv_vol_grams.columns]

        # Combine prices and volumes into a daily table
        daily = piv_price.join(piv_vol_grams).reset_index()
        daily["expiry_date"] = exp

        # Calculate calendar days to expiry for each date
        daily["days_to_expiry"] = (pd.to_datetime(daily["expiry_date"]) - pd.to_datetime(daily["date"])).dt.days

        # Compute percentage gaps between contracts
        # Positive value means the numerator is more expensive per gram
        daily["guinea_vs_ten"] = (daily["GOLDGUINEA"] / daily["GOLDTEN"] - 1.0) * 100.0
        daily["petal_vs_ten"] = (daily["GOLDPETAL"] / daily["GOLDTEN"] - 1.0) * 100.0
        daily["guinea_vs_petal"] = (daily["GOLDGUINEA"] / daily["GOLDPETAL"] - 1.0) * 100.0

        all_daily_gaps.append(daily)

        # -------------------------------------------------------------
        # Step 3: Compute gap summary statistics for this expiry
        # -------------------------------------------------------------
        gap_columns = ["guinea_vs_ten", "petal_vs_ten", "guinea_vs_petal"]
        for gap_col in gap_columns:
            series = daily[gap_col]
            pct_above_zero = (series > 0).mean() * 100.0
            all_gap_stats.append({
                "expiry_date": exp,
                "gap": gap_col,
                "num_days": len(series),
                "mean": round(series.mean(), 4),
                "std": round(series.std(), 4),
                "min": round(series.min(), 4),
                "max": round(series.max(), 4),
                "median": round(series.median(), 4),
                "pct_above_zero": round(pct_above_zero, 2)
            })

        # -------------------------------------------------------------
        # Step 4: Group gaps by days_to_expiry buckets
        # -------------------------------------------------------------
        bucket_bins = [-1, 5, 10, 20, 40, float("inf")]
        bucket_labels = ["0-5", "6-10", "11-20", "21-40", "41+"]
        daily["bucket"] = pd.cut(daily["days_to_expiry"], bins=bucket_bins, labels=bucket_labels)

        for b_label in bucket_labels:
            b_slice = daily[daily["bucket"] == b_label]
            b_count = len(b_slice)
            all_bucket_stats.append({
                "expiry_date": exp,
                "bucket": b_label,
                "num_days": b_count,
                "avg_guinea_vs_ten": round(b_slice["guinea_vs_ten"].mean(), 4) if b_count > 0 else np.nan,
                "avg_petal_vs_ten": round(b_slice["petal_vs_ten"].mean(), 4) if b_count > 0 else np.nan,
                "avg_guinea_vs_petal": round(b_slice["guinea_vs_petal"].mean(), 4) if b_count > 0 else np.nan
            })

    if not all_daily_gaps:
        print("No valid expiries found where all 3 contracts traded together.")
        return

    # Combine all daily records across expiries and save
    daily_gaps_df = pd.concat(all_daily_gaps, ignore_index=True)
    daily_output_cols = [
        "date", "expiry_date", "days_to_expiry",
        "GOLDTEN", "GOLDGUINEA", "GOLDPETAL",
        "guinea_vs_ten", "petal_vs_ten", "guinea_vs_petal"
    ]
    daily_save_df = daily_gaps_df[daily_output_cols].sort_values(by=["expiry_date", "date"]).reset_index(drop=True)
    daily_csv_path = Path(output_folder) / "daily_gaps.csv"
    daily_save_df.to_csv(daily_csv_path, index=False)
    print(f"Saved daily gaps to: {daily_csv_path}")

    # Save and display gap summary statistics
    gap_stats_df = pd.DataFrame(all_gap_stats)
    gap_stats_path = Path(output_folder) / "gap_stats.csv"
    gap_stats_df.to_csv(gap_stats_path, index=False)
    print("\n--- GAP SUMMARY STATISTICS (%) ---")
    print(gap_stats_df.to_string(index=False))

    # Save and display bucket analysis
    gap_bucket_df = pd.DataFrame(all_bucket_stats)
    gap_bucket_path = Path(output_folder) / "gap_by_bucket.csv"
    gap_bucket_df.to_csv(gap_bucket_path, index=False)
    print("\n--- GAPS BY DAYS TO EXPIRY BUCKETS (%) ---")
    print(gap_bucket_df.to_string(index=False))

    # -------------------------------------------------------------
    # Step 5: Liquidity check (Last 10 trading days for each contract & expiry)
    # -------------------------------------------------------------
    liquidity_records = []
    for exp in valid_expiries:
        exp_data = df_filtered[df_filtered["expiry_date"] == exp]
        for sym in target_symbols:
            sym_data = exp_data[exp_data["symbol"] == sym].sort_values(by="date")
            last10 = sym_data.tail(10)
            liquidity_records.append(last10)

    liquidity_df = pd.concat(liquidity_records, ignore_index=True)
    liquidity_cols = ["date", "expiry_date", "symbol", "volume_lots", "grams_per_lot", "volume_grams", "open_interest_lots"]
    liquidity_save_df = liquidity_df[liquidity_cols].reset_index(drop=True)
    liquidity_path = Path(output_folder) / "liquidity_last_days.csv"
    liquidity_save_df.to_csv(liquidity_path, index=False)
    print("\n--- LIQUIDITY CHECK: LAST 10 TRADING DAYS (SAMPLE) ---")
    print(liquidity_save_df.tail(15).to_string(index=False))
    print(f"Full table saved to: {liquidity_path}")

    # -------------------------------------------------------------
    # Step 6: Thin-day check (Low vs High volume based on GOLDGUINEA median)
    # -------------------------------------------------------------
    print("\n--- THIN-DAY CHECK: LOW VS HIGH VOLUME ---")
    for exp in valid_expiries:
        exp_daily = daily_gaps_df[daily_gaps_df["expiry_date"] == exp]
        guinea_median_vol = exp_daily["GOLDGUINEA_vol_grams"].median()

        low_vol = exp_daily[exp_daily["GOLDGUINEA_vol_grams"] <= guinea_median_vol]
        high_vol = exp_daily[exp_daily["GOLDGUINEA_vol_grams"] > guinea_median_vol]

        print(f"Expiry: {exp}")
        print(f"  GOLDGUINEA median daily volume: {guinea_median_vol:,.0f} grams")
        print(f"  Low-Volume Days (<= Median, n={len(low_vol)}): Avg guinea_vs_ten gap = {low_vol['guinea_vs_ten'].mean():.4f}%")
        print(f"  High-Volume Days (> Median, n={len(high_vol)}): Avg guinea_vs_ten gap = {high_vol['guinea_vs_ten'].mean():.4f}%")

    # -------------------------------------------------------------
    # Step 7: Visualizations
    # -------------------------------------------------------------
    num_exp = len(valid_expiries)

    # Chart 1: Gaps over time
    fig, axes = plt.subplots(num_exp, 1, figsize=(12, 5 * num_exp), squeeze=False)
    for idx, exp in enumerate(valid_expiries):
        ax = axes[idx, 0]
        exp_data = daily_gaps_df[daily_gaps_df["expiry_date"] == exp].sort_values(by="date")
        dates = pd.to_datetime(exp_data["date"])

        ax.plot(dates, exp_data["guinea_vs_ten"], label="GOLDGUINEA vs GOLDTEN", color="#1f77b4", lw=2)
        ax.plot(dates, exp_data["petal_vs_ten"], label="GOLDPETAL vs GOLDTEN", color="#ff7f0e", lw=2, linestyle="--")
        ax.plot(dates, exp_data["guinea_vs_petal"], label="GOLDGUINEA vs GOLDPETAL", color="#2ca02c", lw=1.5, alpha=0.8)
        ax.axhline(0.0, color="gray", linestyle=":", lw=1.2, label="Parity (0.0%)")

        ax.set_title(f"Price Spread Gaps Over Time (Expiry: {exp})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Date", fontsize=10)
        ax.set_ylabel("Price Gap (%)", fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart1_path = Path(output_folder) / "gap_over_time.png"
    plt.savefig(chart1_path, dpi=200)
    plt.close()
    print(f"\nSaved chart 1 to: {chart1_path}")

    # Chart 2: Gaps vs Days to Expiry
    fig, axes = plt.subplots(num_exp, 1, figsize=(11, 5 * num_exp), squeeze=False)
    for idx, exp in enumerate(valid_expiries):
        ax = axes[idx, 0]
        exp_data = daily_gaps_df[daily_gaps_df["expiry_date"] == exp].sort_values(by="days_to_expiry")

        ax.scatter(exp_data["days_to_expiry"], exp_data["guinea_vs_ten"], label="GOLDGUINEA vs GOLDTEN", color="#1f77b4", alpha=0.8, s=40)
        ax.scatter(exp_data["days_to_expiry"], exp_data["petal_vs_ten"], label="GOLDPETAL vs GOLDTEN", color="#ff7f0e", alpha=0.8, s=40, marker="^")
        
        # Add smooth trendlines
        z_g = np.polyfit(exp_data["days_to_expiry"], exp_data["guinea_vs_ten"], 1)
        p_g = np.poly1d(z_g)
        ax.plot(exp_data["days_to_expiry"], p_g(exp_data["days_to_expiry"]), color="#1f77b4", linestyle="--", lw=1.5, label="GUINEA vs TEN Trend")

        z_p = np.polyfit(exp_data["days_to_expiry"], exp_data["petal_vs_ten"], 1)
        p_p = np.poly1d(z_p)
        ax.plot(exp_data["days_to_expiry"], p_p(exp_data["days_to_expiry"]), color="#ff7f0e", linestyle="--", lw=1.5, label="PETAL vs TEN Trend")

        ax.axhline(0.0, color="gray", linestyle=":", lw=1.2)
        ax.set_title(f"Price Gaps vs Calendar Days to Expiry (Expiry: {exp})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Days to Expiry (Decreasing Towards Expiry)", fontsize=10)
        ax.set_ylabel("Price Gap (%)", fontsize=10)
        ax.legend(loc="upper left")
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart2_path = Path(output_folder) / "gap_vs_days_to_expiry.png"
    plt.savefig(chart2_path, dpi=200)
    plt.close()
    print(f"Saved chart 2 to: {chart2_path}")

    # -------------------------------------------------------------
    # AT THE END: Plain-English summary (built from real calculated results)
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("PLAIN-ENGLISH SUMMARY")
    print("=" * 75)

    # 1. Per-expiry stats and 0-5 days bucket stats
    for exp in valid_expiries:
        exp_data = daily_gaps_df[daily_gaps_df["expiry_date"] == exp]
        n_days = len(exp_data)
        avg_g = exp_data["guinea_vs_ten"].mean()
        avg_p = exp_data["petal_vs_ten"].mean()
        min_g = exp_data["guinea_vs_ten"].min()
        max_g = exp_data["guinea_vs_ten"].max()
        pct_pos = (exp_data["guinea_vs_ten"] > 0).mean() * 100.0

        exp_b05 = exp_data[exp_data["days_to_expiry"] <= 5]
        if len(exp_b05) > 0:
            b05_mean = exp_b05["guinea_vs_ten"].mean()
            b05_text = f"{b05_mean:.2f}%"
        else:
            b05_text = "no days in 0-5 bucket"

        print(f"- Expiry {exp} ({n_days} days): avg guinea_vs_ten {avg_g:.2f}%, petal_vs_ten {avg_p:.2f}%, guinea range [{min_g:.2f}%, {max_g:.2f}%], {pct_pos:.1f}% days > 0 | last 5 days (0-5 bucket) avg: {b05_text}")

    # 2. Average guinea_vs_petal across all expiries
    avg_gp_all = daily_gaps_df["guinea_vs_petal"].mean()
    print(f"- Average guinea_vs_petal gap across all expiries: {avg_gp_all:.2f}%.")

    # 3. Number of expiries with average guinea_vs_ten > 0
    exp_pos_count = sum(
        daily_gaps_df[daily_gaps_df["expiry_date"] == e]["guinea_vs_ten"].mean() > 0
        for e in valid_expiries
    )
    print(f"- Average guinea_vs_ten gap is above 0 in {exp_pos_count} of {len(valid_expiries)} expiries.")

    # 4. Compare 0-5 bucket vs 41+ bucket across all expiries
    all_b05 = daily_gaps_df[daily_gaps_df["days_to_expiry"] <= 5]["guinea_vs_ten"]
    all_b41 = daily_gaps_df[daily_gaps_df["days_to_expiry"] > 40]["guinea_vs_ten"]
    if len(all_b05) > 0 and len(all_b41) > 0:
        mean_b05 = all_b05.mean()
        mean_b41 = all_b41.mean()
        comp_verb = "smaller" if mean_b05 < mean_b41 else "larger"
        diff = abs(mean_b41 - mean_b05)
        print(f"- Gap near expiry: 0-5 bucket avg ({mean_b05:.2f}%) vs 41+ bucket avg ({mean_b41:.2f}%) is {comp_verb} by {diff:.2f}%.")
    else:
        print("- Gap near expiry: Cannot compare 0-5 and 41+ buckets (insufficient data in one or both buckets).")

    # 5. Caution stating the real number of expiries analyzed
    exp_list_str = ", ".join(valid_expiries)
    print(f"- CAUTION: Analysis is based on only {len(valid_expiries)} expiries ({exp_list_str}); more expiries needed to confirm general persistence.")
    print("=" * 75)


if __name__ == "__main__":
    main()
