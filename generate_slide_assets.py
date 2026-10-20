"""
Generate visual graphic assets for Slide 5 (Market Research / Analysis)
for Canva presentation in Hack in Hills '26.
"""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

os.makedirs('slide_assets', exist_ok=True)

# -------------------------------------------------------------------------
# Asset 1: Target Users Visual Cards (High-res 3-persona banner)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 12)
ax.set_ylim(0, 4.5)
ax.axis('off')

personas = [
    {
        'title': 'Retail & Active Traders',
        'subtitle': 'Contract Selection & Timing',
        'desc': 'Traders choosing between 1g, 8g, 10g & 100g contracts. Need to avoid overpaying on hidden retail premiums.',
        'badge': '1g • 8g • 10g • 100g',
        'color': '#d97706',
        'accent': '#fbbf24',
        'x': 0.5
    },
    {
        'title': 'Small Desks & Prop Traders',
        'subtitle': 'Relative Value Arbitrage',
        'desc': 'Quant & prop desks seeking systematic basis/calendar spreads between contracts. Need cost-adjusted break-evens.',
        'badge': 'Cross-Contract RV Spreads',
        'color': '#2563eb',
        'accent': '#60a5fa',
        'x': 4.3
    },
    {
        'title': 'Analysts & Students',
        'subtitle': 'Reproducible Intelligence',
        'desc': 'Quant researchers and students studying clean MCX bhavcopy data, physical delivery rules & carry models.',
        'badge': 'Open-Source & Rigorous',
        'color': '#059669',
        'accent': '#34d399',
        'x': 8.1
    }
]

for p in personas:
    # Outer card
    card = patches.FancyBboxPatch(
        (p['x'], 0.4), 3.4, 3.7,
        boxstyle="round,pad=0.15,rounding_size=0.25",
        facecolor='#161b22', edgecolor=p['accent'], linewidth=1.5
    )
    ax.add_patch(card)
    
    # Header tag pill
    pill = patches.FancyBboxPatch(
        (p['x'] + 0.2, 3.4), 3.0, 0.45,
        boxstyle="round,pad=0.08,rounding_size=0.15",
        facecolor=p['color'], edgecolor='none', alpha=0.3
    )
    ax.add_patch(pill)
    ax.text(p['x'] + 1.7, 3.62, p['badge'], color=p['accent'], fontsize=10, fontweight='bold', ha='center', va='center')
    
    # Title
    ax.text(p['x'] + 0.2, 3.0, p['title'], color='#ffffff', fontsize=13, fontweight='bold', ha='left', va='center')
    ax.text(p['x'] + 0.2, 2.65, p['subtitle'], color='#94a3b8', fontsize=10, fontstyle='italic', ha='left', va='center')
    
    # Divider line
    ax.plot([p['x'] + 0.2, p['x'] + 3.2], [2.4, 2.4], color='#334155', lw=1)
    
    # Description text
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
    
    ax.text(p['x'] + 0.2, 1.6, desc_formatted, color='#cbd5e1', fontsize=9.5, ha='left', va='center', linespacing=1.4)
    
    # Bottom callout
    use_pill = patches.FancyBboxPatch(
        (p['x'] + 0.2, 0.65), 3.0, 0.4,
        boxstyle="round,pad=0.05,rounding_size=0.1",
        facecolor='#0f172a', edgecolor='#334155', linewidth=1
    )
    ax.add_patch(use_pill)
    ax.text(p['x'] + 1.7, 0.85, "Pre-Trade Fair Value Check", color='#e2e8f0', fontsize=8.5, ha='center', va='center')

plt.tight_layout()
plt.savefig('slide_assets/target_users_personas.png', dpi=300, facecolor='#0d1117', bbox_inches='tight')
plt.close()

# -------------------------------------------------------------------------
# Asset 2: Market Size & Growth Infographic (4 Stat Callouts)
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 3.2), dpi=300)
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')
ax.set_xlim(0, 12)
ax.set_ylim(0, 3.2)
ax.axis('off')

stats = [
    {
        'num': '20.90 Lakh',
        'label': 'Traded Clients on MCX',
        'sub': '+60% YoY from 13L in FY25',
        'color': '#38bdf8',
        'x': 0.4
    },
    {
        'num': '₹5.4 Lakh Cr',
        'label': 'Average Daily Turnover',
        'sub': '+145% YoY surge in FY26',
        'color': '#fbbf24',
        'x': 3.3
    },
    {
        'num': '43%',
        'label': 'Gold Share in Futures',
        'sub': 'Dominant commodity in Q4 FY26',
        'color': '#f43f5e',
        'x': 6.2
    },
    {
        'num': '+496%',
        'label': 'Bullion Turnover Growth',
        'sub': 'Explosive YoY market expansion',
        'color': '#34d399',
        'x': 9.1
    }
]

for s in stats:
    box = patches.FancyBboxPatch(
        (s['x'], 0.3), 2.5, 2.6,
        boxstyle="round,pad=0.1,rounding_size=0.2",
        facecolor='#161b22', edgecolor='#30363d', linewidth=1.2
    )
    ax.add_patch(box)
    
    # Big Stat
    ax.text(s['x'] + 1.25, 2.1, s['num'], color=s['color'], fontsize=18, fontweight='heavy', ha='center', va='center')
    # Label
    ax.text(s['x'] + 1.25, 1.35, s['label'], color='#ffffff', fontsize=10.5, fontweight='bold', ha='center', va='center')
    # Subtitle / Context
    ax.text(s['x'] + 1.25, 0.8, s['sub'], color='#94a3b8', fontsize=8.5, ha='center', va='center')

plt.tight_layout()
plt.savefig('slide_assets/market_size_stats.png', dpi=300, facecolor='#0d1117', bbox_inches='tight')
plt.close()

print("Generated slide visual assets successfully in slide_assets/")
