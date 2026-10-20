"""
Generate light-themed and transparent persona cards matching the
Hack in Hills misty mountain PPT template.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('slide_assets', exist_ok=True)

# -------------------------------------------------------------------------
# Asset 3: Light Theme Target User Personas (Matching Hack in Hills template)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.2), dpi=300)
fig.patch.set_facecolor('#ffffff')
fig.patch.set_alpha(0.0)  # Transparent background
ax.set_facecolor('#ffffff')
ax.patch.set_alpha(0.0)
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.2)
ax.axis('off')

personas_light = [
    {
        'title': 'Active Retail Traders',
        'subtitle': 'Contract Selection & Timing',
        'desc': 'Traders choosing between 1g, 8g, 10g & 100g contracts. Need to avoid overpaying on hidden retail premiums.',
        'badge': '1g • 8g • 10g • 100g',
        'color': '#b45309',
        'accent': '#d97706',
        'bg_card': '#ffffff',
        'pill_bg': '#fef3c7',
        'x': 0.4
    },
    {
        'title': 'Small Desks & Prop Traders',
        'subtitle': 'Relative Value Arbitrage',
        'desc': 'Quant & prop desks seeking systematic basis/calendar spreads between contracts. Need cost-adjusted break-evens.',
        'badge': 'Cross-Contract RV Spreads',
        'color': '#1d4ed8',
        'accent': '#2563eb',
        'bg_card': '#ffffff',
        'pill_bg': '#dbeafe',
        'x': 4.25
    },
    {
        'title': 'Students & Financial Analysts',
        'subtitle': 'Reproducible Intelligence',
        'desc': 'Researchers studying clean MCX bhavcopy data, physical delivery rules, carry decay & open-source code.',
        'badge': 'Reproducible & Rigorous',
        'color': '#047857',
        'accent': '#059669',
        'bg_card': '#ffffff',
        'pill_bg': '#d1fae5',
        'x': 8.1
    }
]

for p in personas_light:
    # Outer card with crisp border and subtle fill
    card = patches.FancyBboxPatch(
        (p['x'], 0.3), 3.5, 3.6,
        boxstyle="round,pad=0.15,rounding_size=0.2",
        facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1.5
    )
    ax.add_patch(card)
    
    # Top Accent Bar
    top_bar = patches.FancyBboxPatch(
        (p['x'], 3.75), 3.5, 0.15,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=p['accent'], edgecolor='none'
    )
    ax.add_patch(top_bar)
    
    # Badge Pill
    pill = patches.FancyBboxPatch(
        (p['x'] + 0.25, 3.15), 3.0, 0.45,
        boxstyle="round,pad=0.05,rounding_size=0.12",
        facecolor=p['pill_bg'], edgecolor=p['color'], linewidth=0.8
    )
    ax.add_patch(pill)
    ax.text(p['x'] + 1.75, 3.37, p['badge'], color=p['color'], fontsize=9.5, fontweight='bold', ha='center', va='center')
    
    # Title & Subtitle
    ax.text(p['x'] + 0.25, 2.75, p['title'], color='#0f172a', fontsize=12.5, fontweight='bold', ha='left', va='center')
    ax.text(p['x'] + 0.25, 2.42, p['subtitle'], color='#64748b', fontsize=9.5, fontstyle='italic', ha='left', va='center')
    
    # Divider
    ax.plot([p['x'] + 0.25, p['x'] + 3.25], [2.2, 2.2], color='#e2e8f0', lw=1)
    
    # Description
    words = p['desc'].split()
    lines = []
    curr = []
    for w in words:
        curr.append(w)
        if len(' '.join(curr)) > 30:
            lines.append(' '.join(curr))
            curr = []
    if curr:
        lines.append(' '.join(curr))
    desc_formatted = '\n'.join(lines)
    
    ax.text(p['x'] + 0.25, 1.45, desc_formatted, color='#334155', fontsize=9.2, ha='left', va='center', linespacing=1.35)
    
    # Use Case Tag
    use_pill = patches.FancyBboxPatch(
        (p['x'] + 0.25, 0.55), 3.0, 0.42,
        boxstyle="round,pad=0.04,rounding_size=0.1",
        facecolor='#f8fafc', edgecolor='#e2e8f0', linewidth=1
    )
    ax.add_patch(use_pill)
    ax.text(p['x'] + 1.75, 0.76, "Use Case: Pre-Trade Fair Value Check", color='#475569', fontsize=8.5, fontweight='semibold', ha='center', va='center')

plt.tight_layout()
plt.savefig('slide_assets/target_users_light_theme.png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Asset 4: Light Theme Market Size & Growth Stats (Matching template)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 2.8), dpi=300)
fig.patch.set_alpha(0.0)
ax.patch.set_alpha(0.0)
ax.set_xlim(0, 12)
ax.set_ylim(0, 2.8)
ax.axis('off')

stats_light = [
    {
        'num': '20.90 Lakh',
        'label': 'Traded Clients on MCX',
        'sub': '+60% YoY from 13L in FY25',
        'color': '#0284c7',
        'x': 0.4
    },
    {
        'num': '₹5.4 Lakh Cr',
        'label': 'Average Daily Turnover',
        'sub': '+145% YoY surge in FY26',
        'color': '#d97706',
        'x': 3.3
    },
    {
        'num': '43%',
        'label': 'Gold Share in Futures',
        'sub': 'Dominant commodity in Q4 FY26',
        'color': '#e11d48',
        'x': 6.2
    },
    {
        'num': '+496%',
        'label': 'Bullion Turnover Growth',
        'sub': 'Explosive YoY bullion expansion',
        'color': '#059669',
        'x': 9.1
    }
]

for s in stats_light:
    box = patches.FancyBboxPatch(
        (s['x'], 0.2), 2.5, 2.4,
        boxstyle="round,pad=0.08,rounding_size=0.18",
        facecolor='#ffffff', edgecolor='#cbd5e1', linewidth=1.2
    )
    ax.add_patch(box)
    
    ax.text(s['x'] + 1.25, 1.85, s['num'], color=s['color'], fontsize=18, fontweight='heavy', ha='center', va='center')
    ax.text(s['x'] + 1.25, 1.2, s['label'], color='#0f172a', fontsize=10.5, fontweight='bold', ha='center', va='center')
    ax.text(s['x'] + 1.25, 0.65, s['sub'], color='#64748b', fontsize=8.5, ha='center', va='center')

plt.tight_layout()
plt.savefig('slide_assets/market_stats_light_theme.png', dpi=300, transparent=True, bbox_inches='tight')
plt.close()

print("Light-theme transparent slide assets generated successfully.")
