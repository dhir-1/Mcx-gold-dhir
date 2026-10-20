# Interactive Dashboard (Streamlit)

This directory contains the interactive web application for the **Commodity Derivatives Intelligence - MCX Gold Futures** project.

---

## What is Inside This Folder?
- **`app.py`**: The Streamlit user interface application. It is strictly dedicated to UI rendering and Plotly visualizations; all computational logic, MCX contract rules, and statistical functions are imported from `core.py`.

---

## Dashboard Features (Tabs)

1. **Today**: Real-time / replay market status banner (Green "No signal today" or Amber "Signal") and active expiry reference comparison tables.
2. **Gap monitor**: Interactive time-series tracking of percentage gaps for retail contracts (GOLDTEN, GOLDGUINEA, GOLDPETAL) against the reference benchmark, including shaded historical normal levels ($\pm$ break-even bounds).
3. **Settlement**: Due Date Rate (DDR) physical settlement convergence analysis, showing how screen trading premiums vanish at final settlement.
4. **Backtest**: Pairs trading backtest trade logs with gross returns, net after exchange/brokerage fees, and net after slippage.
5. **Calendar & liquidity**: Gantt chart of contract trading lifecycles, tender periods (last 3 trading days with 8.34% tender margin), no-new-entries periods, and physical volume turnover in grams.
6. **Method & limits**: Full MCX contract specifications table, complete assumption list, and analytical boundaries.

---

## How to Run

### From the Repository Root:
```bash
streamlit run streamlit/app.py
```
*(Alternatively: `streamlit run app.py`)*

### From Inside This Directory:
```bash
cd streamlit
streamlit run app.py
```

Access the dashboard in your web browser at: `http://localhost:8501`
