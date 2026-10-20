"""
app.py - Commodity Derivatives Intelligence Dashboard
Streamlit UI only. All calculations and business logic reside in core.py.
Run with: streamlit run app.py
"""

import os
import sys

# Ensure repository root is on sys.path whether run from root or inside subdirectories
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT_DIR = os.path.abspath(os.path.join(_CURRENT_DIR, ".."))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import core

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & THEME
# Clean light theme with Gold (#C98A1B) accent and Navy (#2F4558) text.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MCX Gold Futures Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom light styling
st.markdown(
    f"""
    <style>
    .main {{
        background-color: #FAFAFA;
        color: {core.COLOR_TEXT_NAVY};
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }}
    h1, h2, h3, h4 {{
        color: {core.COLOR_TEXT_NAVY};
        font-weight: 700;
    }}
    .gold-pill {{
        background-color: #FEF3C7;
        color: #B45309;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
        font-size: 0.85rem;
        display: inline-block;
        border: 1px solid #FCD34D;
    }}
    .caption-box {{
        background-color: #F1F5F9;
        border-left: 4px solid {core.COLOR_ACCENT_GOLD};
        padding: 8px 12px;
        margin-top: 6px;
        margin-bottom: 14px;
        border-radius: 0 4px 4px 0;
        font-size: 0.88rem;
        color: {core.COLOR_TEXT_NAVY};
    }}
    .status-banner-signal {{
        background-color: #FEF3C7;
        border-left: 6px solid #D97706;
        padding: 14px 18px;
        border-radius: 4px;
        color: #92400E;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 16px;
    }}
    .status-banner-ok {{
        background-color: #D1FAE5;
        border-left: 6px solid #059669;
        padding: 14px 18px;
        border-radius: 4px;
        color: #065F46;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 16px;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# DATA LOADING (Safe, non-crashing)
# -----------------------------------------------------------------------------
@st.cache_data
def get_data():
    df_clean = core.load_clean_data("out/gold_futures_clean.csv")
    df_ddr = core.load_ddr_data("results_ddr/ddr.csv")
    df_trades = core.load_trades_data("results_bt/trades.csv")
    return df_clean, df_ddr, df_trades


df_clean, df_ddr, df_trades = get_data()

# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS
# Replay date selector, assumed cost break-even, and model window parameters.
# -----------------------------------------------------------------------------
st.sidebar.title("🪙 MCX Gold Intelligence")
st.sidebar.markdown(
    f"<span style='color:{core.COLOR_ACCENT_GOLD}; font-weight:bold;'>Gold contract price comparison</span>",
    unsafe_allow_html=True,
)

if df_clean.empty:
    st.error("⚠️ Clean dataset missing: `out/gold_futures_clean.csv`. Run `preprocess.py` first.")
    st.stop()

# Available trading dates for replay
available_dates = sorted(df_clean["date"].unique())
latest_date = available_dates[-1]

as_of_date = st.sidebar.date_input(
    "📅 As-of Date (Replay Mode)",
    value=latest_date.date(),
    min_value=available_dates[0].date(),
    max_value=latest_date.date(),
    help="Filters all calculations to records on or before this date to prevent look-ahead bias.",
)
as_of_ts = pd.Timestamp(as_of_date)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Model Parameters")
st.sidebar.caption("Fixed benchmark defaults per strategy specification:")

break_even_pct = st.sidebar.number_input(
    "Assumed Break-Even Cost (%)",
    min_value=0.05,
    max_value=1.00,
    value=float(core.DEFAULT_BREAK_EVEN_PCT),
    step=0.01,
    format="%.2f",
    help="Assumed round-trip costs including CTT, exchange turnover, stamp duty, brokerage, and slippage.",
)

rolling_window = st.sidebar.number_input(
    "Normal Level Window (Trading Days)",
    min_value=3,
    max_value=30,
    value=int(core.DEFAULT_ROLLING_WINDOW),
    step=1,
    help="Number of past trading days used to compute the historical normal level (no look-ahead).",
)

cutoff_days = st.sidebar.number_input(
    "Expiry Cutoff (Calendar Days)",
    min_value=1,
    max_value=30,
    value=int(core.DEFAULT_EXPIRY_CUTOFF_DAYS),
    step=1,
    help="Positions are not entered if days to expiry is below this threshold due to liquidity rollover.",
)

st.sidebar.markdown("---")
st.sidebar.info(
    "📌 **Core Rule**: Parameters are fixed benchmarks to prevent overfitting on small commodity samples."
)

# Header
st.title("Commodity Derivatives Intelligence")
st.markdown(
    f"Comparing **GOLDM (100g, 995)** with **GOLDTEN (10g)**, **GOLDGUINEA (8g)**, and **GOLDPETAL (1g)** (all 999 purity) on MCX."
)

# -----------------------------------------------------------------------------
# TABS SETUP
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "1 Today",
    "2 Gap monitor",
    "3 Settlement",
    "4 Backtest",
    "5 Calendar & liquidity",
    "6 Method & limits",
])

# =============================================================================
# TAB 1: TODAY / STATUS BANNER
# =============================================================================
with tabs[0]:
    st.subheader(f"Current Market Status as of {as_of_ts.strftime('%d-%b-%Y')}")

    banner, tables = core.get_today_status_and_tables(
        df_clean,
        as_of_ts,
        window=rolling_window,
        break_even_pct=break_even_pct,
        cutoff_days=cutoff_days,
    )

    if banner["code"] == "SIGNAL":
        st.markdown(
            f"<div class='status-banner-signal'>🟠 {banner['message']}</div>",
            unsafe_allow_html=True,
        )
    elif banner["code"] == "NO_DATA":
        st.markdown(
            f"<div class='status-banner-signal' style='background-color:#E2E8F0; border-left:6px solid #64748B; color:#334155;'>ℹ️ {banner['message']} — The selected date has no traded bhavcopy records (weekend or exchange holiday).</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='status-banner-ok'>🟢 {banner['message']} — All active contracts are within normal relative ranges.</div>",
            unsafe_allow_html=True,
        )

    if banner["code"] == "NO_DATA":
        st.info(f"No trading data available for {as_of_ts.strftime('%d-%b-%Y')}. Please choose an active trading day in the sidebar.")
    elif not tables:
        st.info(f"No active 999 contracts found with trades as of {as_of_ts.strftime('%d-%b-%Y')}.")
    else:
        for t_info in tables:
            st.markdown(f"#### Active Expiry: {t_info['expiry']}")
            st.markdown(
                f"**Reference Benchmark**: `{t_info['reference_desc']}` &nbsp;|&nbsp; "
                f"**Benchmark Price**: `₹{t_info['reference_price']:,.2f} / g`"
            )

            tbl = t_info["table"].copy()
            # Styling helper supporting both map (pandas >= 2.1) and applymap
            format_dict = {
                "Price (₹/g)": "₹{:,.2f}",
                "Gap vs Ref (%)": "{:+.3f}%",
                "Normal Level (%)": "{:+.3f}%",
                "Excess Over Normal (%)": "{:+.3f}%",
                "Break-Even (assumed %)": "{:.2f}%",
                "Days to Expiry": "{:d}",
            }
            if "Reference Used" in tbl.columns:
                format_dict["Reference Used"] = lambda s: str(s)

            styler = tbl.style.format(format_dict)
            if hasattr(styler, "map"):
                styler = styler.map(
                    lambda v: "color: #b91c1c; font-weight: bold;" if "Signal" in str(v) else "",
                    subset=["Status"]
                )
            else:
                styler = styler.applymap(
                    lambda v: "color: #b91c1c; font-weight: bold;" if "Signal" in str(v) else "",
                    subset=["Status"]
                )
            st.dataframe(
                styler,
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("---")

    st.markdown(
        "<div class='caption-box'><b>Takeaway:</b> Provides an immediate pre-trade check of whether retail gold contracts are trading at a fair relative value or whether an actionable spread anomaly has emerged.</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 2: GAP MONITOR
# =============================================================================
with tabs[1]:
    st.subheader("Historical Gap Monitor vs Reference Benchmark")

    # Available 999 expiries in the dataset
    all_999_expiries = sorted(
        df_clean[df_clean["symbol"].isin(["GOLDTEN", "GOLDGUINEA", "GOLDPETAL"])]["expiry_date"].unique()
    )
    expiry_options = [e.strftime("%Y-%m-%d") for e in all_999_expiries]

    selected_exp_str = st.selectbox(
        "Select Contract Expiry to Monitor:",
        options=expiry_options,
        index=len(expiry_options) - 1,
    )
    selected_exp = pd.Timestamp(selected_exp_str)

    gaps_df, ref_desc = core.compute_expiry_gaps(
        df_clean,
        selected_exp,
        as_of_date=as_of_ts,
        window=rolling_window,
        break_even_pct=break_even_pct,
        cutoff_days=cutoff_days,
    )

    if gaps_df.empty:
        st.warning(f"No traded gap records available for expiry {selected_exp_str} up to {as_of_ts.strftime('%Y-%m-%d')}.")
    else:
        st.markdown(f"**Reference Benchmark for {selected_exp_str}**: `{ref_desc}`")

        # Plotly multi-line chart
        fig = go.Figure()

        colors = {
            "GOLDTEN": "#2563EB",     # Blue
            "GOLDGUINEA": "#059669",   # Green
            "GOLDPETAL": "#DC2626",    # Red
        }

        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#475569",
            annotation_text="Benchmark (0.0%)",
            annotation_position="bottom right",
        )

        for sym in gaps_df["symbol"].unique():
            sub = gaps_df[gaps_df["symbol"] == sym].sort_values("date")
            fig.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["gap_pct"],
                mode="lines+markers",
                name=f"{sym} Gap (%)",
                line=dict(color=colors.get(sym, "#6B7280"), width=2),
                marker=dict(size=4),
            ))

            # Shaded break-even band around normal level
            fig.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["normal_level"] + break_even_pct,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["normal_level"] - break_even_pct,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(201, 138, 27, 0.12)",
                name=f"{sym} Normal Band (±{break_even_pct}%)",
            ))

        fig.update_layout(
            title=f"Price Gap (%) vs Reference for Expiry {selected_exp_str}",
            xaxis_title="Trading Date",
            yaxis_title="Price Gap (%)",
            template="plotly_white",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"<div class='caption-box'><b>Takeaway:</b> Tracks the percentage premium of each contract over the reference benchmark. Shaded gold ribbons illustrate the 10-day rolling normal level ± {break_even_pct}% assumed round-trip costs. Breakthroughs beyond the band indicate potential mispricings.</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 3: SETTLEMENT (DDR)
# =============================================================================
with tabs[2]:
    st.subheader("Settlement Convergence: Last Traded Close vs Due Date Rate (DDR)")

    if df_ddr.empty:
        st.warning("⚠️ Due Date Rate reference table `results_ddr/ddr.csv` is missing.")
    else:
        df_conv = core.compute_settlement_convergence(df_clean, df_ddr)

        if df_conv.empty:
            st.info("No matching settlement records found.")
        else:
            col_l, col_r = st.columns([1, 1])

            with col_l:
                st.markdown("#### DDR Difference Table (₹/g and %)")
                display_conv = df_conv[[
                    "symbol",
                    "expiry_date",
                    "last_trade_date",
                    "last_close_per_gram",
                    "ddr_per_gram",
                    "diff_rupees_per_gram",
                    "diff_pct",
                ]].copy()
                display_conv.rename(columns={
                    "symbol": "Contract",
                    "expiry_date": "Expiry",
                    "last_trade_date": "Final Trade Date",
                    "last_close_per_gram": "Last Close (₹/g)",
                    "ddr_per_gram": "DDR (₹/g)",
                    "diff_rupees_per_gram": "Diff (₹/g)",
                    "diff_pct": "Diff (%)",
                }, inplace=True)

                st.dataframe(
                    display_conv.style.format({
                        "Last Close (₹/g)": "₹{:,.2f}",
                        "DDR (₹/g)": "₹{:,.2f}",
                        "Diff (₹/g)": "{:+,.2f}",
                        "Diff (%)": "{:+.3f}%",
                    }),
                    use_container_width=True,
                    hide_index=True,
                )

            with col_r:
                fig_bar = px.bar(
                    df_conv,
                    x="symbol",
                    y="diff_pct",
                    color="expiry_date",
                    barmode="group",
                    title="Convergence Discrepancy at Final Settlement (%)",
                    labels={"symbol": "Contract", "diff_pct": "Difference vs DDR (%)", "expiry_date": "Expiry"},
                    color_discrete_sequence=[core.COLOR_ACCENT_GOLD, "#2563EB"],
                )
                fig_bar.update_layout(template="plotly_white", margin=dict(l=30, r=30, t=50, b=30))
                st.plotly_chart(fig_bar, use_container_width=True)

            st.info(
                "📌 **Settlement Mechanics Note**: DDR is the final settlement price, not an executable price. "
                "GOLDM settles on a 3-day average of spot prices; the other three settle on the spot price on expiry day, "
                "so their settlement prices are not computed the same way."
            )
            st.markdown(
                "<div class='caption-box'><b>Takeaway:</b> The extra price paid while trading largely disappears at final settlement: GOLDGUINEA ends at about 0.00% from DDR and GOLDPETAL within about 0.01% to 0.26%.</div>",
                unsafe_allow_html=True,
            )

# =============================================================================
# TAB 4: BACKTEST
# =============================================================================
with tabs[3]:
    st.subheader("Simple Pairs Arbitrage Backtest (GOLDTEN vs GOLDPETAL)")

    if df_trades.empty:
        st.warning("⚠️ Backtest results file `results_bt/trades.csv` not found.")
    else:
        # Determine period classification
        df_show_trades = df_trades.copy()
        if "period" not in df_show_trades.columns:
            df_show_trades["period"] = df_show_trades["entry_date"].apply(
                lambda d: "Design period" if str(d)[:10] <= core.DESIGN_PERIOD_CUTOFF
                else ("Unseen test period" if str(d)[:10] >= core.UNSEEN_PERIOD_START else "Intermediate")
            )

        st.markdown("#### Realized Trade Log")
        st.dataframe(
            df_show_trades[[
                "strategy",
                "expiry",
                "period",
                "entry_date",
                "exit_date",
                "days_held",
                "gap_at_entry",
                "gap_at_exit",
                "gross_pnl",
                "fees",
                "slippage",
                "net_pnl_fees",
                "net_pnl_fees_slippage",
            ]].style.format({
                "entry_date": lambda d: str(d)[:10],
                "exit_date": lambda d: str(d)[:10],
                "expiry": lambda d: str(d)[:10],
                "gap_at_entry": "{:.3f}%",
                "gap_at_exit": "{:.3f}%",
                "gross_pnl": "₹{:,.2f}",
                "fees": "₹{:,.2f}",
                "slippage": "₹{:,.2f}",
                "net_pnl_fees": "₹{:,.2f}",
                "net_pnl_fees_slippage": "₹{:,.2f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("---")
        chart_img_path = core._resolve_data_path("results_bt/strategy_vs_gold.png")
        if os.path.exists(chart_img_path):
            st.image(
                chart_img_path,
                caption="Backtest Performance: Strategy Cumulative P&L vs Standalone Long Gold",
                use_container_width=True,
            )

        st.warning("⚠️ **Model Caution**: Costs are assumed, sample is small (3 expiries analyzed). Past performance does not guarantee future results.")
        st.markdown(
            "<div class='caption-box'><b>Takeaway:</b> Gross results were mixed (some trades gained, some lost) and no reliable profit remained after assumed costs. Sample is small.</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 5: CALENDAR & LIQUIDITY
# =============================================================================
with tabs[4]:
    st.subheader("Contract Lifecycle, Tender Periods & Physical Turnover")

    contracts_info, df_vol = core.compute_calendar_timeline(df_clean, df_trades)

    if contracts_info:
        st.markdown("#### Contract Lifecycle Timeline")

        timeline_rows = []
        for c in contracts_info:
            c_name = f"{c['symbol']} ({c['expiry_str']})"
            timeline_rows.append(dict(Contract=c_name, Start=c["first_trade"], End=c["last_trade"], Stage="Active Trading"))
            timeline_rows.append(dict(Contract=c_name, Start=c["no_entry_start"], End=c["expiry"], Stage="No New Entries (Last 10d)"))
            timeline_rows.append(dict(Contract=c_name, Start=c["tender_start"], End=c["expiry"], Stage="Tender Period (3d, 8.34% Margin)"))

        df_gantt = pd.DataFrame(timeline_rows)
        fig_timeline = px.timeline(
            df_gantt,
            x_start="Start",
            x_end="End",
            y="Contract",
            color="Stage",
            color_discrete_map={
                "Active Trading": "#94A3B8",
                "No New Entries (Last 10d)": "#F59E0B",
                "Tender Period (3d, 8.34% Margin)": "#DC2626",
            },
            title="Contract Lifecycle & Regulatory Restrictions",
        )
        fig_timeline.update_yaxes(autorange="reversed")
        fig_timeline.update_layout(template="plotly_white", margin=dict(l=40, r=40, t=50, b=40))
        st.plotly_chart(fig_timeline, use_container_width=True)

        # Show note for estimated tender period on running contracts
        if any(c.get("tender_is_estimated", False) for c in contracts_info):
            st.caption("ℹ️ Tender period for running contracts is estimated from weekdays only. Exchange holidays are not accounted for.")

    st.markdown("#### Physical Gold Turnover Comparison")
    if not df_vol.empty:
        vol_styler = df_vol.style.format({
            "Avg Daily Lots": "{:,.1f}",
            "Avg Daily Volume (g)": "{:,.1f} g",
            "Avg Daily Volume (kg)": "{:,.2f} kg",
        })
        if hasattr(vol_styler, "map"):
            vol_styler = vol_styler.map(
                lambda v: "color: #b91c1c; font-weight: bold;" if "Thin" in str(v) else "",
                subset=["Liquidity Flag"]
            )
        else:
            vol_styler = vol_styler.applymap(
                lambda v: "color: #b91c1c; font-weight: bold;" if "Thin" in str(v) else "",
                subset=["Liquidity Flag"]
            )
        st.dataframe(
            vol_styler,
            use_container_width=True,
            hide_index=True,
        )

    st.caption("⚠️ **Liquidity Caveat**: Volume is not market depth. Bids and asks may be thin or wide even on traded days.")
    st.markdown(
        "<div class='caption-box'><b>Takeaway:</b> Highlights key delivery restrictions (tender periods requiring compulsory physical delivery margins) and confirms that GOLDM accounts for the vast majority of physical gold turnover.</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# TAB 6: METHOD & LIMITS
# =============================================================================
with tabs[5]:
    st.subheader("Methodology, Assumptions & Contract Specifications")

    st.markdown("""
    ### 🔬 Calculation Methodology
    1. **Purity Normalization**:
       * GOLDM is standardized from $995$ to $999$ purity using the official MCX conversion factor:
         $$\\text{Price}_{999} = \\text{Price}_{995} \\times \\frac{999}{995}$$
       * GOLDTEN, GOLDGUINEA, and GOLDPETAL are already $999$ fineness and require no purity conversion.
    2. **Calendar Expiry Blending**:
       * Surrounding GOLDM contracts ($E_1 \\le E < E_2$) are interpolated to match the $999$ contract expiry $E$:
         $$w_2 = \\frac{E - E_1}{E_2 - E_1}, \\quad w_1 = 1 - w_2, \\quad \\text{Blend} = w_1 \\cdot P_{E_1} + w_2 \\cdot P_{E_2}$$
       * Filtered to exclude thin rollover days (where $E_1$ has $\\le 7$ calendar days remaining).
    3. **10-Day Rolling Normal Level**:
       * The historical baseline is computed strictly using prior trading days (excluding today) via `shift(1).rolling(10).mean()` to guarantee zero look-ahead bias.
    4. **Signal Trigger Bound**:
       * A spread anomaly triggers only when $|\\text{Gap} - \\text{Normal Level}| > \\text{Break-Even}$ and $\\text{Days to Expiry} > 10$.
    """)

    st.markdown("### 📋 MCX Contract Specifications Reference")
    specs_list = []
    for sym, sp in core.CONTRACT_SPECS.items():
        row = dict(Product=sym, **sp)
        specs_list.append(row)
    df_specs_display = pd.DataFrame(specs_list)
    st.dataframe(df_specs_display, use_container_width=True, hide_index=True)
    st.caption("*Source: MCX contract specifications.*")

    st.markdown("### ⚠️ Analytical Assumptions & Limitations")
    st.markdown("""
    * **Assumed Trading Friction**: Baseline round-trip cost assumed at **0.22%** (covering CTT, exchange turnover fees, SEBI charges, stamp duty, and brokerage).
    * **Slippage**: Market impact and bid-ask slippage are assumed at benchmark rates; real fills depend on live limit book depth.
    * **Due Date Rates (DDR)**: Hand-collected from official MCX Bhavcopy reports; DDR represents a polled spot settlement benchmark, not an executable quote.
    * **Settlement Prices vs Execution**: Exchange Bhavcopy prices are official daily settlement figures, not live quotes.
    * **Sample Limitation**: Analysis is evaluated on an empirical sample of 3 historical contract expiries.
    * **Physical Delivery Omission**: Compulsory physical delivery costs, vaulting fees, and making charges are not modelled in trading P&L.
    * **Data Source**: Daily Bhavcopy published by the Multi Commodity Exchange of India (MCX).
    """)

    st.markdown(
        "<div class='caption-box'><b>Takeaway:</b> Rigorous documentation of assumptions and contract rules ensures that theoretical pricing differences are grounded in exchange institutional reality.</div>",
        unsafe_allow_html=True,
    )
