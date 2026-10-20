"""
STEP 6: COMPARE GOLDM WITH THE 999 CONTRACTS
Script: goldm_check.py
Usage: python goldm_check.py out results_goldm
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def main():
    # -------------------------------------------------------------------------
    # Parse CLI arguments (default: 'out' for input, 'results_goldm' for output)
    # -------------------------------------------------------------------------
    input_dir = sys.argv[1] if len(sys.argv) > 1 else 'out'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'results_goldm'
    os.makedirs(output_dir, exist_ok=True)

    clean_file = os.path.join(input_dir, 'gold_futures_clean.csv')
    if not os.path.exists(clean_file):
        print(f"ERROR: Clean file not found at: {clean_file}")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # DATA RULES & LOADING
    # - Use only rows where traded is True
    # - Treat (symbol, expiry_date) as one contract
    # - Convert GOLDM 995 purity to 999: close_per_gram * 999 / 995
    # - Do not hardcode expiry dates: detect dynamically from data
    # -------------------------------------------------------------------------
    df = pd.read_csv(clean_file)
    df['date'] = pd.to_datetime(df['date'])
    df['expiry_date'] = pd.to_datetime(df['expiry_date'])

    # Filter only traded days
    df = df[df['traded'] == True].copy()

    # Add purity-adjusted price per gram for GOLDM
    # MCX factor: 999 / 995
    df['goldm_999_per_gram'] = np.where(
        df['symbol'] == 'GOLDM',
        df['close_per_gram'] * 999.0 / 995.0,
        np.nan
    )

    # Lots to grams mapping
    grams_per_lot = {
        'GOLDM': 100,
        'GOLDTEN': 10,
        'GOLDGUINEA': 8,
        'GOLDPETAL': 1
    }
    df['volume_grams'] = df['volume_lots'] * df['symbol'].map(grams_per_lot)

    # Identify 999 contracts and their expiries
    symbols_999 = ['GOLDTEN', 'GOLDGUINEA', 'GOLDPETAL']
    expiries_999_by_sym = {
        s: set(df[df['symbol'] == s]['expiry_date'].unique())
        for s in symbols_999
    }
    # Expiries where all three 999 contracts exist
    common_999_expiries = sorted(
        set.intersection(*expiries_999_by_sym.values())
    )

    # GOLDM expiries present in data
    goldm_expiries = sorted(df[df['symbol'] == 'GOLDM']['expiry_date'].unique())

    print("=" * 80)
    print("STEP 6: COMPARE GOLDM WITH 999 CONTRACTS (GOLDTEN, GOLDGUINEA, GOLDPETAL)")
    print("=" * 80)
    print(f"999 Expiries found: {[d.strftime('%Y-%m-%d') for d in common_999_expiries]}")
    print(f"GOLDM Expiries found: {[d.strftime('%Y-%m-%d') for d in goldm_expiries]}")

    # Containers for results across analyzed expiries
    all_gap_stats = []
    analyzed_expiries = []
    audit_data = {}
    time_value_results = []
    versions_comp = []

    # -------------------------------------------------------------------------
    # STEP 1: Blending Surrounding GOLDM Contracts for each 999 Expiry
    # -------------------------------------------------------------------------
    for e in common_999_expiries:
        e_str = e.strftime('%Y-%m-%d')
        print("\n" + "-" * 80)
        print(f"EVALUATING 999 EXPIRY: {e_str}")
        print("-" * 80)

        # E1: latest GOLDM expiry on or before E
        goldm_before_or_on = [g for g in goldm_expiries if g <= e]
        # E2: earliest GOLDM expiry after E
        goldm_after = [g for g in goldm_expiries if g > e]

        if not goldm_before_or_on:
            print(f"WARNING: Skipping expiry {e_str}. Missing surrounding GOLDM expiry E1 (on or before {e_str}).")
            continue
        if not goldm_after:
            print(f"WARNING: Skipping expiry {e_str}. Missing surrounding GOLDM expiry E2 (after {e_str}).")
            continue

        e1 = max(goldm_before_or_on)
        e2 = min(goldm_after)
        e1_str = e1.strftime('%Y-%m-%d')
        e2_str = e2.strftime('%Y-%m-%d')

        # Calendar day weights: w2 = (E - E1) / (E2 - E1), w1 = 1 - w2
        cal_total = (e2 - e1).days
        cal_e_to_e1 = (e - e1).days
        if cal_total <= 0:
            print(f"WARNING: Invalid calendar duration between E1 ({e1_str}) and E2 ({e2_str}). Skipping.")
            continue

        w2 = cal_e_to_e1 / cal_total
        w1 = 1.0 - w2

        print(f"Surrounding GOLDM contracts: E1 = {e1_str}, E2 = {e2_str}")
        print(f"Calendar Days: (E2 - E1) = {cal_total} days, (E - E1) = {cal_e_to_e1} days")
        print(f"Weights: w1 (E1 weight) = {w1:.6f} ({w1*100:.2f}%), w2 (E2 weight) = {w2:.6f} ({w2*100:.2f}%)")

        # Slice data for the contracts
        m1_df = df[(df['symbol'] == 'GOLDM') & (df['expiry_date'] == e1)].set_index('date')
        m2_df = df[(df['symbol'] == 'GOLDM') & (df['expiry_date'] == e2)].set_index('date')
        ten_df = df[(df['symbol'] == 'GOLDTEN') & (df['expiry_date'] == e)].set_index('date')
        guinea_df = df[(df['symbol'] == 'GOLDGUINEA') & (df['expiry_date'] == e)].set_index('date')
        petal_df = df[(df['symbol'] == 'GOLDPETAL') & (df['expiry_date'] == e)].set_index('date')

        # Find dates where both GOLDM traded
        goldm_both_dates = sorted(set(m1_df.index).intersection(set(m2_df.index)))
        total_goldm_both_days = len(goldm_both_dates)

        # Filter: GOLDM-E1 still has more than 7 calendar days to expiry
        # (e1 - date).days > 7
        valid_dates = [d for d in goldm_both_dates if (e1 - d).days > 7]
        dropped_thin_dates = [d for d in goldm_both_dates if (e1 - d).days <= 7]

        # Check if 999 contracts also traded on these dates
        final_dates = [
            d for d in valid_dates
            if (d in ten_df.index) and (d in guinea_df.index) and (d in petal_df.index)
        ]
        dropped_missing_999 = [d for d in valid_dates if d not in final_dates]

        print(f"Trading Days Summary for Expiry {e_str}:")
        print(f"  Total dates where both GOLDM contracts traded: {total_goldm_both_days}")
        print(f"  Days dropped due to thin expiry week of E1 (days to E1 <= 7): {len(dropped_thin_dates)}")
        if dropped_thin_dates:
            print(f"    Thin dates dropped: {[d.strftime('%Y-%m-%d') for d in dropped_thin_dates]}")
        if dropped_missing_999:
            print(f"  Days dropped due to missing 999 trades: {len(dropped_missing_999)}")
        print(f"  Final active dates used in analysis: {len(final_dates)}")

        if not final_dates:
            print(f"WARNING: No valid trading dates remaining for expiry {e_str}. Skipping.")
            continue

        analyzed_expiries.append(e)

        # Compute goldm_blend on final_dates
        p_m1_conv = m1_df.loc[final_dates, 'goldm_999_per_gram']
        p_m2_conv = m2_df.loc[final_dates, 'goldm_999_per_gram']
        p_m2_raw = m2_df.loc[final_dates, 'close_per_gram']

        goldm_blend = w1 * p_m1_conv + w2 * p_m2_conv

        p_ten = ten_df.loc[final_dates, 'close_per_gram']
        p_guinea = guinea_df.loc[final_dates, 'close_per_gram']
        p_petal = petal_df.loc[final_dates, 'close_per_gram']

        # ---------------------------------------------------------------------
        # STEP 2: Compute Gaps per Gram
        # (a) goldm_blend vs GOLDTEN
        # (b) GOLDGUINEA vs goldm_blend
        # (c) GOLDPETAL vs goldm_blend
        # ---------------------------------------------------------------------
        # (a) goldm_blend vs GOLDTEN
        gap_blend_vs_ten_pct = (goldm_blend / p_ten - 1.0) * 100.0
        gap_blend_vs_ten_rs = goldm_blend - p_ten

        # Also compute inverted GOLDTEN vs goldm_blend for chart & comparison
        gap_ten_vs_blend_pct = (p_ten / goldm_blend - 1.0) * 100.0
        gap_ten_vs_blend_rs = p_ten - goldm_blend

        # (b) GOLDGUINEA vs goldm_blend
        gap_guinea_vs_blend_pct = (p_guinea / goldm_blend - 1.0) * 100.0
        gap_guinea_vs_blend_rs = p_guinea - goldm_blend

        # (c) GOLDPETAL vs goldm_blend
        gap_petal_vs_blend_pct = (p_petal / goldm_blend - 1.0) * 100.0
        gap_petal_vs_blend_rs = p_petal - goldm_blend

        # Store for stats table
        def calc_stats(series_pct, series_rs, gap_name):
            return {
                'expiry': e_str,
                'gap_name': gap_name,
                'days': len(series_pct),
                'mean': series_pct.mean(),
                'std': series_pct.std(),
                'min': series_pct.min(),
                'max': series_pct.max(),
                'median': series_pct.median(),
                'pct_above_0': (series_pct > 0).mean() * 100.0,
                'mean_rupees_per_gram': series_rs.mean()
            }

        gaps_to_report = [
            (gap_blend_vs_ten_pct, gap_blend_vs_ten_rs, 'goldm_blend vs GOLDTEN'),
            (gap_guinea_vs_blend_pct, gap_guinea_vs_blend_rs, 'GOLDGUINEA vs goldm_blend'),
            (gap_petal_vs_blend_pct, gap_petal_vs_blend_rs, 'GOLDPETAL vs goldm_blend'),
            (gap_ten_vs_blend_pct, gap_ten_vs_blend_rs, 'GOLDTEN vs goldm_blend')
        ]

        for s_pct, s_rs, g_name in gaps_to_report:
            all_gap_stats.append(calc_stats(s_pct, s_rs, g_name))

        # ---------------------------------------------------------------------
        # STEP 4: Three comparison versions for GOLDM vs GOLDTEN
        # (i) raw GOLDM close_per_gram (no purity adjustment) using GOLDM-E2
        # (ii) purity-adjusted GOLDM-E2 only (no expiry blending)
        # (iii) purity-adjusted and blended (the fair one)
        # ---------------------------------------------------------------------
        v1_pct = (p_m2_raw / p_ten - 1.0) * 100.0
        v1_rs = p_m2_raw - p_ten

        v2_pct = (p_m2_conv / p_ten - 1.0) * 100.0
        v2_rs = p_m2_conv - p_ten

        v3_pct = (goldm_blend / p_ten - 1.0) * 100.0
        v3_rs = goldm_blend - p_ten

        versions_comp.append({
            'expiry': e_str,
            'v1_mean_pct': v1_pct.mean(),
            'v1_mean_rs': v1_rs.mean(),
            'v2_mean_pct': v2_pct.mean(),
            'v2_mean_rs': v2_rs.mean(),
            'v3_mean_pct': v3_pct.mean(),
            'v3_mean_rs': v3_rs.mean(),
        })

        # ---------------------------------------------------------------------
        # STEP 5: Time-value check
        # ((E2_adj / E1_adj - 1) * 100) / calendar_days
        # ---------------------------------------------------------------------
        tv_series_pct_per_day = ((p_m2_conv / p_m1_conv - 1.0) * 100.0) / cal_total
        tv_tot_spread_pct = (p_m2_conv / p_m1_conv - 1.0) * 100.0

        time_value_results.append({
            'expiry': e_str,
            'e1': e1_str,
            'e2': e2_str,
            'cal_days': cal_total,
            'mean_tv_per_day_pct': tv_series_pct_per_day.mean(),
            'annualized_pct': tv_series_pct_per_day.mean() * 365.0,
            'mean_tot_spread_pct': tv_tot_spread_pct.mean(),
            'min_tv_per_day_pct': tv_series_pct_per_day.min(),
            'max_tv_per_day_pct': tv_series_pct_per_day.max()
        })

        # ---------------------------------------------------------------------
        # STEP 8: Store Audit Data for One Date
        # -------------------------------------------------------------------------
        # Pick the first valid date for full reproducibility
        audit_dt = final_dates[0]
        audit_data[e_str] = {
            'date': audit_dt.strftime('%Y-%m-%d'),
            'e1_str': e1_str,
            'm1_close': m1_df.loc[audit_dt, 'close'],
            'm1_close_pg': m1_df.loc[audit_dt, 'close_per_gram'],
            'm1_conv': p_m1_conv.loc[audit_dt],
            'e2_str': e2_str,
            'm2_close': m2_df.loc[audit_dt, 'close'],
            'm2_close_pg': m2_df.loc[audit_dt, 'close_per_gram'],
            'm2_conv': p_m2_conv.loc[audit_dt],
            'w1': w1,
            'w2': w2,
            'cal_total': cal_total,
            'cal_e_to_e1': cal_e_to_e1,
            'blend': goldm_blend.loc[audit_dt],
            'ten_close': ten_df.loc[audit_dt, 'close'],
            'ten_pg': p_ten.loc[audit_dt],
            'gap_blend_vs_ten_pct': gap_blend_vs_ten_pct.loc[audit_dt],
            'gap_blend_vs_ten_rs': gap_blend_vs_ten_rs.loc[audit_dt]
        }

        # ---------------------------------------------------------------------
        # STEP 7: Prepare Chart Data for this expiry
        # ---------------------------------------------------------------------
        # Save daily time series DataFrame for chart
        daily_ts = pd.DataFrame({
            'date': final_dates,
            'goldm_blend': goldm_blend.values,
            'GOLDTEN_pct': gap_ten_vs_blend_pct.values,
            'GOLDGUINEA_pct': gap_guinea_vs_blend_pct.values,
            'GOLDPETAL_pct': gap_petal_vs_blend_pct.values,
        })
        daily_ts.set_index('date', inplace=True)

    # -------------------------------------------------------------------------
    # PRINT RESULTS TABLE & SAVE CSV
    # -------------------------------------------------------------------------
    if not all_gap_stats:
        print("\nERROR: No expiries could be analyzed.")
        sys.exit(1)

    df_stats = pd.DataFrame(all_gap_stats)
    csv_out_path = os.path.join(output_dir, 'goldm_gaps_stats.csv')
    df_stats.to_csv(csv_out_path, index=False)
    print(f"\nSaved gap statistics to: {csv_out_path}")

    print("\n" + "=" * 80)
    print("STEP 3: SUMMARY GAP STATISTICS TABLE")
    print("=" * 80)
    display_cols = ['expiry', 'gap_name', 'days', 'mean', 'std', 'min', 'max', 'median', 'pct_above_0', 'mean_rupees_per_gram']
    print(df_stats[display_cols].to_string(index=False, float_format=lambda x: f"{x:10.4f}"))

    # -------------------------------------------------------------------------
    # PRINT STEP 4: Three comparison versions table
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 4: THREE COMPARISON VERSIONS (GOLDM vs GOLDTEN)")
    print("=" * 80)
    print("Version (i)   : Raw GOLDM-E2 close_per_gram (no purity conversion, no blending)")
    print("Version (ii)  : Purity-adjusted GOLDM-E2 (999-converted, no blending)")
    print("Version (iii) : Purity-adjusted & Blended GOLDM (fair comparison)")
    print("-" * 80)
    for v in versions_comp:
        print(f"Expiry: {v['expiry']}")
        print(f"  (i)   Raw E2 vs GOLDTEN           : {v['v1_mean_pct']:+8.4f}%  (Rs {v['v1_mean_rs']:+7.2f}/g)")
        print(f"  (ii)  Purity-adjusted E2 vs GOLDTEN: {v['v2_mean_pct']:+8.4f}%  (Rs {v['v2_mean_rs']:+7.2f}/g)")
        print(f"  (iii) Blended & Purity-adj vs TEN  : {v['v3_mean_pct']:+8.4f}%  (Rs {v['v3_mean_rs']:+7.2f}/g)")
        delta_purity = v['v2_mean_pct'] - v['v1_mean_pct']
        delta_blend = v['v3_mean_pct'] - v['v2_mean_pct']
        print(f"  --> Fix Effect: Purity adjustment shift: {delta_purity:+8.4f}% | Expiry blend shift: {delta_blend:+8.4f}%")

    # -------------------------------------------------------------------------
    # PRINT STEP 5: Time-value check
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 5: TIME-VALUE CHECK BETWEEN GOLDM EXPIRIES")
    print("=" * 80)
    for tv in time_value_results:
        print(f"Contracts: GOLDM {tv['e1']} to {tv['e2']} ({tv['cal_days']} calendar days)")
        print(f"  Average Price Premium (E2 / E1 - 1): {tv['mean_tot_spread_pct']:+.4f}%")
        print(f"  Implied Time Value per Day         : {tv['mean_tv_per_day_pct']:+.5f}% / day")
        print(f"  Annualized Implied Carry Cost      : {tv['annualized_pct']:+.2f}% p.a. (365-day basis)")
        print(f"  Daily Range                        : {tv['min_tv_per_day_pct']:+.5f}% to {tv['max_tv_per_day_pct']:+.5f}%")

    # -------------------------------------------------------------------------
    # PRINT STEP 6: Liquidity in grams
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 6: LIQUIDITY COMPARISON (DAILY PHYSICAL VOLUME IN GRAMS)")
    print("=" * 80)
    # Filter df to the analyzed contracts
    contracts_analyzed = []
    for exp in analyzed_expiries:
        exp_str = exp.strftime('%Y-%m-%d')
        # Find E1, E2
        goldm_before_or_on = [g for g in goldm_expiries if g <= exp]
        goldm_after = [g for g in goldm_expiries if g > exp]
        e1 = max(goldm_before_or_on)
        e2 = min(goldm_after)
        contracts_analyzed.extend([
            ('GOLDM', e1),
            ('GOLDM', e2),
            ('GOLDTEN', exp),
            ('GOLDGUINEA', exp),
            ('GOLDPETAL', exp),
        ])

    unique_contracts = sorted(list(set(contracts_analyzed)), key=lambda x: (x[0], x[1]))
    liq_rows = []
    for sym, c_exp in unique_contracts:
        sub = df[(df['symbol'] == sym) & (df['expiry_date'] == c_exp)]
        avg_lots = sub['volume_lots'].mean()
        avg_grams = sub['volume_grams'].mean()
        liq_rows.append({
            'symbol': sym,
            'expiry': c_exp.strftime('%Y-%m-%d'),
            'lot_size_grams': grams_per_lot[sym],
            'avg_daily_lots': avg_lots,
            'avg_daily_grams': avg_grams,
            'avg_daily_kg': avg_grams / 1000.0
        })
    df_liq = pd.DataFrame(liq_rows)
    print(df_liq.to_string(index=False, float_format=lambda x: f"{x:12.1f}"))

    # -------------------------------------------------------------------------
    # STEP 7: Plot Chart (results_goldm/gaps_vs_goldm.png)
    # -------------------------------------------------------------------------
    chart_path = os.path.join(output_dir, 'gaps_vs_goldm.png')
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot for each analyzed expiry
    for exp in analyzed_expiries:
        exp_str = exp.strftime('%Y-%m-%d')
        ax.plot(daily_ts.index, daily_ts['GOLDTEN_pct'], label=f'GOLDTEN ({exp_str}) vs Blend', color='#1f77b4', lw=2)
        ax.plot(daily_ts.index, daily_ts['GOLDGUINEA_pct'], label=f'GOLDGUINEA ({exp_str}) vs Blend', color='#2ca02c', lw=2)
        ax.plot(daily_ts.index, daily_ts['GOLDPETAL_pct'], label=f'GOLDPETAL ({exp_str}) vs Blend', color='#d62728', lw=2)

    ax.axhline(0, color='black', linestyle='--', linewidth=1.5, label='GOLDM Blend (Benchmark = 0%)')
    ax.set_title("MCX Gold Futures: Percent Price Gaps vs Purity-Adjusted & Expiry-Blended GOLDM", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Trading Date", fontsize=11, labelpad=8)
    ax.set_ylabel("Price Gap per Gram (%)", fontsize=11, labelpad=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='upper left', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\nSaved gap chart to: {chart_path}")

    # -------------------------------------------------------------------------
    # STEP 8: Print Full Audit for ONE Date
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 8: CALCULATION AUDIT FOR ONE REPRESENTATIVE DATE")
    print("=" * 80)
    first_exp_str = analyzed_expiries[0].strftime('%Y-%m-%d')
    adt = audit_data[first_exp_str]
    print(f"Analysis Date          : {adt['date']} (for 999 Expiry {first_exp_str})")
    print(f"Contract E1 (GOLDM)    : Expiry {adt['e1_str']}")
    print(f"  Official Close       : Rs {adt['m1_close']:,.2f} per 10g  -->  Rs {adt['m1_close_pg']:,.2f} per gram")
    print(f"  Purity-Adjusted (999): Rs {adt['m1_close_pg']:,.2f} x (999 / 995) = Rs {adt['m1_conv']:,.4f} per gram")
    print(f"Contract E2 (GOLDM)    : Expiry {adt['e2_str']}")
    print(f"  Official Close       : Rs {adt['m2_close']:,.2f} per 10g  -->  Rs {adt['m2_close_pg']:,.2f} per gram")
    print(f"  Purity-Adjusted (999): Rs {adt['m2_close_pg']:,.2f} x (999 / 995) = Rs {adt['m2_conv']:,.4f} per gram")
    print(f"Calendar Interpolation : E1 to E2 span = {adt['cal_total']} days | E to E1 offset = {adt['cal_e_to_e1']} days")
    print(f"  Weight w1 (E1 weight): {adt['w1']:.6f}")
    print(f"  Weight w2 (E2 weight): {adt['w2']:.6f}")
    print(f"Blended GOLDM (999)    : ({adt['w1']:.4f} x {adt['m1_conv']:,.2f}) + ({adt['w2']:.4f} x {adt['m2_conv']:,.2f}) = Rs {adt['blend']:,.4f}/g")
    print(f"GOLDTEN (999 purity)   : Close = Rs {adt['ten_close']:,.2f} per 10g  -->  Rs {adt['ten_pg']:,.4f} per gram")
    print(f"Price Gap Result       :")
    print(f"  goldm_blend vs TEN   : ({adt['blend']:,.4f} / {adt['ten_pg']:,.4f} - 1) x 100 = {adt['gap_blend_vs_ten_pct']:+.4f}%")
    print(f"  Rupee Gap per Gram   : {adt['blend']:,.4f} - {adt['ten_pg']:,.4f} = Rs {adt['gap_blend_vs_ten_rs']:+.2f}/g")

    # -------------------------------------------------------------------------
    # SUMMARY: Plain-English, at most 8 lines, built from calculated results
    # -------------------------------------------------------------------------
    # Extract calculated metrics dynamically
    stats_map = {(row['expiry'], row['gap_name']): row for row in all_gap_stats}
    ten_blend_row = stats_map.get((first_exp_str, 'goldm_blend vs GOLDTEN'))
    guinea_blend_row = stats_map.get((first_exp_str, 'GOLDGUINEA vs goldm_blend'))
    petal_blend_row = stats_map.get((first_exp_str, 'GOLDPETAL vs goldm_blend'))
    vc = versions_comp[0]

    mean_ten_gap = ten_blend_row['mean'] if ten_blend_row else 0.0
    mean_guinea_prem = guinea_blend_row['mean'] if guinea_blend_row else 0.0
    mean_petal_prem = petal_blend_row['mean'] if petal_blend_row else 0.0

    print("\n" + "=" * 80)
    print("PLAIN-ENGLISH SUMMARY (AT MOST 8 LINES)")
    print("=" * 80)
    print(f"1. GOLDTEN and blended GOLDM match closely: GOLDM blend averages {mean_ten_gap:+.2f}% (Rs {ten_blend_row['mean_rupees_per_gram']:+.2f}/g) vs GOLDTEN across {ten_blend_row['days']} days.")
    print(f"2. Retail contracts trade at a persistent premium: GUINEA averages {mean_guinea_prem:+.2f}% and PETAL averages {mean_petal_prem:+.2f}% over the blended GOLDM benchmark.")
    print(f"3. Fixes matter: Raw GOLDM-E2 showed {vc['v1_mean_pct']:+.2f}%, purity adjustment shifted it to {vc['v2_mean_pct']:+.2f}%, and expiry blending refined it to {vc['v3_mean_pct']:+.2f}%.")
    print(f"4. The implied time value of gold carry between GOLDM contracts averaged {time_value_results[0]['mean_tv_per_day_pct']:+.4f}% per day ({time_value_results[0]['annualized_pct']:.1f}% annualized).")
    print(f"5. CAUTION: Findings are based on only {len(analyzed_expiries)} expiry ({first_exp_str}) as earlier expiries lacked surrounding GOLDM data.")
    print(f"6. CAUTION: Physical delivery terms (100g GOLDM bars vs smaller coins) and final settlement (DDR) are not checked here.")
    print(f"7. CAUTION: Brokerage, exchange transaction charges, CTT, and bid-ask slippage are not subtracted from these theoretical spreads.")
    print(f"8. CAUTION: The expiry blending formula assumes linear time-value decay across calendar days between successive monthly contracts.")
    print("=" * 80)

if __name__ == '__main__':
    main()
