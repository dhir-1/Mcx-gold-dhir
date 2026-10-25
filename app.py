"""
app.py - Commodity Derivatives Intelligence Dashboard
Streamlit UI only. All calculations and business logic reside in core.py.
Run with: streamlit run app.py
"""

import os
import sys

# Ensure repository root is on sys.path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
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
# Premium dark theme with Gold (#D4AF37) accent and elevated slate surfaces.
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="MCX Gold Futures Intelligence",
    page_icon="🪙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dark styling
st.markdown(
    """
    <style>
    /* Content container sizing and spacing */
    .block-container {
        max-width: 1240px;
        padding-top: 1.8rem;
        padding-bottom: 2.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        margin: 0 auto;
    }

    /* Top header */
    .dashboard-header {
        margin-bottom: 1.5rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #2D3748;
    }
    .dashboard-title {
        font-size: 2.1rem;
        font-weight: 700;
        color: #F1F5F9;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
    }
    .dashboard-subtitle {
        font-size: 1.15rem;
        font-weight: 600;
        color: #D4AF37;
        margin-bottom: 0.35rem;
    }
    .dashboard-meta {
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.5;
    }

    /* Tab navigation styling */
    .stTabs [role="tablist"] {
        gap: 8px;
        border-bottom: 1px solid #2D3748;
        padding-bottom: 2px;
    }
    .stTabs [role="tab"] {
        padding: 8px 16px;
        border-radius: 6px 6px 0 0;
        font-weight: 500;
        color: #94A3B8;
        border: 1px solid transparent;
        background-color: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [role="tab"]:hover {
        color: #F1F5F9;
        background-color: rgba(212, 175, 55, 0.06);
    }
    .stTabs [role="tab"][aria-selected="true"] {
        color: #D4AF37;
        border-bottom: 2px solid #D4AF37;
        font-weight: 600;
    }

    /* Metric card styling */
    div[data-testid="stMetric"] {
        background-color: #1A2234;
        border: 1px solid #2D3748;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.25);
    }
    div[data-testid="stMetric"] label {
        font-size: 0.78rem;
        font-weight: 500;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.3rem;
        font-weight: 700;
        color: #F1F5F9;
    }

    /* Status banner styling (calm colors, non-alarming) */
    .status-banner-ok {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.12), rgba(6, 78, 59, 0.25));
        border: 1px solid #059669;
        border-left: 5px solid #10B981;
        padding: 14px 18px;
        border-radius: 8px;
        color: #A7F3D0;
        font-size: 0.98rem;
        font-weight: 500;
        margin-bottom: 18px;
    }
    .status-banner-signal {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.12), rgba(120, 53, 15, 0.25));
        border: 1px solid #D97706;
        border-left: 5px solid #F59E0B;
        padding: 14px 18px;
        border-radius: 8px;
        color: #FDE68A;
        font-size: 0.98rem;
        font-weight: 500;
        margin-bottom: 18px;
    }
    .status-banner-nodata {
        background: linear-gradient(135deg, rgba(100, 116, 139, 0.12), rgba(30, 41, 59, 0.35));
        border: 1px solid #475569;
        border-left: 5px solid #64748B;
        padding: 14px 18px;
        border-radius: 8px;
        color: #CBD5E1;
        font-size: 0.98rem;
        font-weight: 500;
        margin-bottom: 18px;
    }

    /* Takeaway / Caption Box */
    .caption-box {
        background-color: #1A2234;
        border: 1px solid #2D3748;
        border-left: 4px solid #D4AF37;
        padding: 12px 16px;
        margin-top: 14px;
        margin-bottom: 16px;
        border-radius: 0 8px 8px 0;
        font-size: 0.9rem;
        line-height: 1.5;
        color: #CBD5E1;
    }

    /* Dataframe container rounded border */
    [data-testid="stDataFrame"] {
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid #2D3748;
    }

    /* Responsive adjustments */
    @media (max-width: 768px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }
        .dashboard-title {
            font-size: 1.6rem;
        }
    }
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
    "<span style='color:#D4AF37; font-weight:bold;'>Gold contract price comparison</span>",
    unsafe_allow_html=True,
)

if df_clean.empty:
    st.error("⚠️ Clean dataset missing: `out/gold_futures_clean.csv`. Run `scripts/preprocess.py` first.")
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
st.markdown(
    """
    <div class="dashboard-header">
        <div class="dashboard-title">🪙 Commodity Derivatives Intelligence</div>
        <div class="dashboard-subtitle">Gold contract price comparison</div>
        <div class="dashboard-meta">
            Replay on saved MCX data, 3 Aug to 1 Oct 2026 &nbsp;•&nbsp; 
            Comparing <b>GOLDM (100g, 995)</b> with <b>GOLDTEN (10g)</b>, <b>GOLDGUINEA (8g)</b>, and <b>GOLDPETAL (1g)</b> (all 999 purity) on MCX
        </div>
    </div>
    """,
    unsafe_allow_html=True,
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

    # Key metrics strip at top of tab
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("As-of Date", as_of_ts.strftime('%d-%b-%Y'))
    with m2:
        status_label = "Signal" if banner["code"] == "SIGNAL" else ("No Data" if banner["code"] == "NO_DATA" else "Normal")
        st.metric("Market Status", status_label)
    with m3:
        st.metric("Assumed Cost", f"{break_even_pct:.2f}%")
    with m4:
        st.metric("Normal Window", f"{rolling_window} Days")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if banner["code"] == "SIGNAL":
        st.markdown(
            f"<div class='status-banner-signal'>⚡ {banner['message']}</div>",
            unsafe_allow_html=True,
        )
    elif banner["code"] == "NO_DATA":
        st.markdown(
            f"<div class='status-banner-nodata'>ℹ️ {banner['message']} — The selected date has no traded bhavcopy records (weekend or exchange holiday).</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='status-banner-ok'>✓ {banner['message']} — All active contracts are within normal relative ranges.</div>",
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
            status_style_fn = lambda v: "color: #F59E0B; font-weight: 600;" if "Signal" in str(v) else "color: #10B981;"
            if hasattr(styler, "map"):
                styler = styler.map(status_style_fn, subset=["Status"])
            else:
                styler = styler.applymap(status_style_fn, subset=["Status"])
            st.dataframe(
                styler,
                use_container_width=True,
                hide_index=True,
            )
            st.markdown("---")

    st.markdown(
        "<div class='caption-box'><b>Takeaway:</b> Provides an immediate pre-trade check of whether retail gold contracts are trading at a fair relative value or whether a gap bigger than the assumed cost has emerged.</div>",
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

    # Key metrics strip at top of tab
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Monitored Expiry", selected_exp_str)
    with m2:
        st.metric("Benchmark", ref_desc.split(" ")[0] if ref_desc else "GOLDM")
    with m3:
        st.metric("Normal Window", f"{rolling_window} Days")
    with m4:
        st.metric("Cost Band", f"±{break_even_pct:.2f}%")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    if gaps_df.empty:
        st.warning(f"No traded gap records available for expiry {selected_exp_str} up to {as_of_ts.strftime('%Y-%m-%d')}.")
    else:
        st.markdown(f"**Reference Benchmark for {selected_exp_str}**: `{ref_desc}`")

        # Plotly multi-line chart
        fig = go.Figure()

        colors = {
            "GOLDTEN": "#60A5FA",     # Soft bright blue
            "GOLDGUINEA": "#34D399",   # Fresh emerald green
            "GOLDPETAL": "#F87171",    # Soft coral red
        }

        # Add zero line
        fig.add_hline(
            y=0,
            line_dash="dash",
            line_color="#94A3B8",
            annotation_text="Benchmark (0.0%)",
            annotation_position="bottom right",
            annotation_font=dict(color="#94A3B8", size=10),
        )

        for sym in gaps_df["symbol"].unique():
            sub = gaps_df[gaps_df["symbol"] == sym].sort_values("date")
            fig.add_trace(go.Scatter(
                x=sub["date"],
                y=sub["gap_pct"],
                mode="lines+markers",
                name=f"{sym} Gap (%)",
                line=dict(color=colors.get(sym, "#A78BFA"), width=2),
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
                fillcolor="rgba(212, 175, 55, 0.16)",
                name=f"{sym} Normal Band (±{break_even_pct}%)",
            ))

        fig.update_layout(
            title=dict(
                text=f"Price Gap (%) vs Reference for Expiry {selected_exp_str}",
                font=dict(size=14, color="#F1F5F9"),
            ),
            xaxis_title="Trading Date",
            yaxis_title="Price Gap (%)",
            template="plotly_dark",
            paper_bgcolor="#1A2234",
            plot_bgcolor="#1A2234",
            hovermode="x unified",
            height=430,
            font=dict(family="sans serif", color="#F1F5F9", size=11),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10),
            ),
        )
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#2D3748")
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#2D3748")

        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            f"<div class='caption-box'><b>Takeaway:</b> Tracks the percentage premium of each contract over the reference benchmark. Shaded gold ribbons illustrate the 10-day rolling normal level ± {break_even_pct}% assumed round-trip costs. Breakthroughs beyond the band indicate unusual price gaps exceeding break-even costs.</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 3: SETTLEMENT (DDR)
# =============================================================================
with tabs[2]:
    st.subheader("Settlement Price Comparison: Last Traded Close vs Due Date Rate (DDR)")

    if df_ddr.empty:
        st.warning("⚠️ Due Date Rate reference table `results_ddr/ddr.csv` is missing.")
    else:
        df_conv = core.compute_settlement_convergence(df_clean, df_ddr)

        if df_conv.empty:
            st.info("No matching settlement records found.")
        else:
            # Key metrics strip at top of tab
            n_exp = df_conv["expiry_date"].nunique()
            n_contracts = len(df_conv)
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Expiries Analyzed", f"{n_exp} Expiries ({n_contracts} contracts)")
            with m2:
                st.metric("DDR Relative Range", "0.00% to -0.26%")
            with m3:
                st.metric("Last Close vs DDR", "+0.44% to +1.02%")
            with m4:
                st.metric("Settlement Basis", "Exchange Spot Rate")

            st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

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
                    title="Settlement Difference vs DDR (%)",
                    labels={"symbol": "Contract", "diff_pct": "Difference vs DDR (%)", "expiry_date": "Expiry"},
                    color_discrete_sequence=["#D4AF37", "#60A5FA"],
                )
                fig_bar.update_layout(
                    template="plotly_dark",
                    paper_bgcolor="#1A2234",
                    plot_bgcolor="#1A2234",
                    height=420,
                    font=dict(family="sans serif", color="#F1F5F9", size=11),
                    margin=dict(l=30, r=30, t=50, b=30),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font=dict(size=10),
                    ),
                )
                fig_bar.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#2D3748")
                fig_bar.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#2D3748")
                st.plotly_chart(fig_bar, use_container_width=True)

            st.info(
                "📌 **Settlement Mechanics Note**: DDR is the final settlement price, not an executable price. "
                "GOLDM settles on a 3-day average of spot prices; the other three settle on the spot price on expiry day, "
                "so their settlement prices are not computed the same way."
            )
            st.markdown(
                "<div class='caption-box'><b>Takeaway:</b> The final settlement prices (DDR) of GOLDGUINEA and GOLDPETAL are within 0.00% and -0.01% to -0.26% of GOLDTEN's DDR, because they settle on the same exchange spot price. However, the last traded close was still 0.44% to 1.02% above the contract's own DDR.</div>",
                unsafe_allow_html=True,
            )

# =============================================================================
# TAB 4: BACKTEST
# =============================================================================
with tabs[3]:
    st.subheader("Simple Pairs Trading Backtest (GOLDTEN vs GOLDPETAL)")

    if df_trades.empty:
        st.warning("⚠️ Backtest results file `results_bt/trades.csv` not found.")
    else:
        # Key metrics strip at top of tab
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Total Trades", "5 Trades")
        with m2:
            st.metric("Gross Result", "+₹1,556")
        with m3:
            st.metric("Friction (Fees & Slippage)", "-₹1,633")
        with m4:
            st.metric("Net Result", "-₹77")

        st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

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
            "<div class='caption-box'><b>Takeaway:</b> Five simple trades over 3 expiries made a small gross profit (Rs 1,556), but fees and slippage removed it (net Rs -77). No reliable profit.</div>",
            unsafe_allow_html=True,
        )

# =============================================================================
# TAB 5: CALENDAR & LIQUIDITY
# =============================================================================
with tabs[4]:
    st.subheader("Contract Lifecycle, Tender Periods & Physical Turnover")

    contracts_info, df_vol = core.compute_calendar_timeline(df_clean, df_trades)

    # Key metrics strip at top of tab
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Turnover Leader", "GOLDM (~99%)")
    with m2:
        st.metric("Active Contracts", "4 Variants")
    with m3:
        st.metric("Tender Period", "Last 3 Trading Days")
    with m4:
        st.metric("Compulsory Margin", "8.34% per day")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

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
                "Active Trading": "#64748B",
                "No New Entries (Last 10d)": "#D4AF37",
                "Tender Period (3d, 8.34% Margin)": "#F87171",
            },
            title="Contract Lifecycle & Regulatory Restrictions",
        )
        fig_timeline.update_yaxes(autorange="reversed", showgrid=True, gridwidth=1, gridcolor="#2D3748")
        fig_timeline.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#2D3748")
        fig_timeline.update_layout(
            template="plotly_dark",
            paper_bgcolor="#1A2234",
            plot_bgcolor="#1A2234",
            height=460,
            font=dict(family="sans serif", color="#F1F5F9", size=11),
            margin=dict(l=40, r=40, t=50, b=40),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10),
            ),
        )
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
        vol_style_fn = lambda v: "color: #F59E0B; font-weight: 600;" if "Thin" in str(v) else ""
        if hasattr(vol_styler, "map"):
            vol_styler = vol_styler.map(vol_style_fn, subset=["Liquidity Flag"])
        else:
            vol_styler = vol_styler.applymap(vol_style_fn, subset=["Liquidity Flag"])
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

    # Key metrics strip at top of tab
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Purity Conversion", "995 → 999 (× 999/995)")
    with m2:
        st.metric("Normal Window", "10-Day Lagged Mean")
    with m3:
        st.metric("Assumed Cost", "0.22% Round-Trip")
    with m4:
        st.metric("Expiry Cutoff", "≤ 10 Calendar Days")

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

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
    * **Slippage**: Market impact and bid-ask slippage are assumed at benchmark rates; real fills depend on order book depth.
    * **Due Date Rates (DDR)**: Hand-collected from official MCX Bhavcopy reports; DDR represents a polled spot settlement benchmark, not an executable quote.
    * **Settlement Prices vs Execution**: Exchange Bhavcopy prices are official daily settlement figures, not intraday quotes.
    * **Sample Limitation**: Analysis is evaluated on an empirical sample of 3 historical contract expiries.
    * **Physical Delivery Omission**: Compulsory physical delivery costs, vaulting fees, and making charges are not modelled in trading P&L.
    * **Data Source**: Daily Bhavcopy published by the Multi Commodity Exchange of India (MCX).
    """)

    st.markdown(
        "<div class='caption-box'><b>Takeaway:</b> Rigorous documentation of assumptions and contract rules ensures that theoretical pricing differences are grounded in exchange institutional reality.</div>",
        unsafe_allow_html=True,
    )
