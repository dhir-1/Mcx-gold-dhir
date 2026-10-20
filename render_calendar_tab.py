"""
Render high-resolution visual snapshot of the Calendar & Liquidity tab.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.dates as mdates

import core

os.makedirs('results', exist_ok=True)

df_clean = core.load_clean_data("out/gold_futures_clean.csv")
df_trades = core.load_trades_data("results_bt/trades.csv")
contracts_info, df_vol = core.compute_calendar_timeline(df_clean, df_trades)

fig = plt.figure(figsize=(14, 11), dpi=250)
fig.patch.set_facecolor('#FAFAFA')

# Top section: Header & Gantt Timeline
ax1 = plt.subplot2grid((11, 1), (0, 0), rowspan=6)
ax1.set_facecolor('#FFFFFF')

# Color map matching Streamlit UI
color_trading = '#94A3B8'
color_no_entry = '#F59E0B'
color_tender = '#DC2626'

y_labels = []
y_ticks = []

for idx, c in enumerate(contracts_info):
    y_pos = idx
    c_name = f"{c['symbol']} ({c['expiry_str']})"
    y_labels.append(c_name)
    y_ticks.append(y_pos)

    t0 = mdates.date2num(c['first_trade'])
    t1 = mdates.date2num(c['last_trade'])
    exp = mdates.date2num(c['expiry'])
    no_entry = mdates.date2num(c['no_entry_start'])
    tender = mdates.date2num(c['tender_start'])

    # 1. Active Trading bar (from first trade to last trade or expiry)
    ax1.barh(y_pos, t1 - t0, left=t0, height=0.45, color=color_trading, alpha=0.9, label='Active Trading' if idx == 0 else "")

    # 2. No New Entries (Last 10 calendar days before expiry)
    ax1.barh(y_pos, exp - no_entry, left=no_entry, height=0.45, color=color_no_entry, alpha=0.9, label='No New Entries (Last 10d)' if idx == 0 else "")

    # 3. Tender Period (Last 3 trading days)
    ax1.barh(y_pos, exp - tender, left=tender, height=0.45, color=color_tender, alpha=0.95, label='Tender Period (3d, 8.34% Margin)' if idx == 0 else "")

ax1.set_yticks(y_ticks)
ax1.set_yticklabels(y_labels, fontsize=10, fontweight='bold', color=core.COLOR_TEXT_NAVY)
ax1.invert_yaxis()  # top down
ax1.xaxis_date()
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b-%Y'))
ax1.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
plt.setp(ax1.get_xticklabels(), rotation=25, ha='right', fontsize=9, color=core.COLOR_TEXT_NAVY)

ax1.set_title("Contract Lifecycle & Regulatory Restrictions (MCX Gold Futures)", fontsize=13, fontweight='bold', pad=12, color=core.COLOR_TEXT_NAVY)
ax1.grid(True, linestyle=':', alpha=0.5, axis='x')
ax1.legend(loc='upper right', framealpha=0.95, fontsize=9.5)

# Add Note under chart
has_estimated = any(c.get("tender_is_estimated", False) for c in contracts_info)
if has_estimated:
    ax1.text(0.01, -0.15, "ℹ️ Tender period for running contracts is estimated from weekdays only. Exchange holidays are not accounted for.",
             transform=ax1.transAxes, fontsize=9.5, fontstyle='italic', color='#475569')

# Bottom section: Physical Gold Turnover Table
ax2 = plt.subplot2grid((11, 1), (7, 0), rowspan=4)
ax2.set_facecolor('#FAFAFA')
ax2.axis('off')

ax2.text(0.0, 1.0, "Physical Gold Turnover Comparison", fontsize=12, fontweight='bold', color=core.COLOR_TEXT_NAVY)

# Format table data
table_data = []
table_headers = ["Contract", "Expiry", "Lot Size", "Avg Daily Lots", "Avg Daily Volume (g)", "Avg Daily Volume (kg)", "Liquidity Flag"]

for _, row in df_vol.iterrows():
    table_data.append([
        row["Contract"],
        row["Expiry"],
        f"{row['Lot Size (g)']} g",
        f"{row['Avg Daily Lots']:,.1f}",
        f"{row['Avg Daily Volume (g)']:,.1f} g",
        f"{row['Avg Daily Volume (kg)']:,.2f} kg",
        row["Liquidity Flag"]
    ])

tbl = ax2.table(
    cellText=table_data,
    colLabels=table_headers,
    cellLoc='center',
    loc='center',
    bbox=[0.0, 0.12, 1.0, 0.78]
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)

# Style header and rows
for (r, c), cell in tbl.get_celld().items():
    if r == 0:
        cell.set_facecolor('#F1F5F9')
        cell.set_text_props(fontweight='bold', color=core.COLOR_TEXT_NAVY)
    else:
        if "Thin" in table_data[r-1][6]:
            cell.set_facecolor('#FEF2F2')
            if c == 6:
                cell.set_text_props(color='#B91C1C', fontweight='bold')
        else:
            cell.set_facecolor('#FFFFFF' if r % 2 == 0 else '#F8FAFC')
    cell.set_edgecolor('#E2E8F0')

ax2.text(0.0, -0.05, "⚠️ Liquidity Caveat: Volume is not market depth. Bids and asks may be thin or wide even on traded days.\nTakeaway: Highlights key delivery restrictions and confirms that GOLDM accounts for the vast majority of physical gold turnover.",
         fontsize=9, fontstyle='italic', color='#64748B')

plt.tight_layout()
out_path = 'results/calendar_tab_screenshot.png'
plt.savefig(out_path, dpi=250, facecolor='#FAFAFA', bbox_inches='tight')
plt.close()
print(f"Calendar tab screenshot saved successfully to: {out_path}")
