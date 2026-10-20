# Commodity Derivatives Intelligence - MCX Gold Futures
**(Hack in Hills '26, Team Lexa, Dhir Agrawal)**

---

## The Problem
MCX (Multi Commodity Exchange of India) lists the same physical gold under four different contract sizes:
- **GOLDM** (100 grams, 995 purity, quotes in Rs per 10g)
- **GOLDTEN** (10 grams, 999 purity, quotes in Rs per 10g)
- **GOLDGUINEA** (8 grams, 999 purity, quotes in Rs per 8g)
- **GOLDPETAL** (1 gram, 999 purity, quotes in Rs per 1g)

Raw screen prices cannot be compared directly because:
- **Lot sizes** and quotation units differ across products.
- **Purity standards** differ (GOLDM is 995 fineness, whereas the other three are 999 fineness).
- **Expiry dates** differ (GOLDM expires on the 5th of the month, whereas the other three expire on the last calendar day).
- **Settlement rules** differ (GOLDM settles on a 3-day polled average spot price; the other three settle on the expiry day's spot Due Date Rate).

---

## What the Tool Does
1. **Normalizes Pricing**: Converts all contract prices to a standardized, like-for-like price-per-gram in 999-purity equivalent terms (applying the official MCX conversion factor of $\times 999 / 995$ to GOLDM).
2. **Calendar Blending**: Interpolates calendar-matched GOLDM reference prices when expiries do not align.
3. **Relative-Value Gap Tracking**: Computes percentage gaps of retail contracts (GOLDTEN, GOLDGUINEA, GOLDPETAL) against the benchmark reference.
4. **Noise Filtering**: Evaluates rolling 10-day historical normal levels without look-ahead bias and stays quiet unless an observed gap exceeds assumed round-trip trading friction (default 0.22%) and has more than 10 days to expiry.

---

## Early Findings
*(Empirical observations from the dataset)*
- While trading, **GOLDGUINEA** and **GOLDPETAL** cost about **0.43% more per gram** than the benchmark, on all 40 days checked.
- At final settlement (DDR), the gap almost disappears: **GOLDGUINEA** ends at about **0.00%**, and **GOLDPETAL** within **-0.01% to -0.26%**.
- Simple trading strategies showed **no reliable profit after assumed costs**.

---

## Limits & Caveats
- **Sample Window**: Data covers 3 Aug – 1 Oct 2026 only.
- **Settlement Sample**: Only 2 contract expiries have observed final settlement prices (DDR).
- **Assumed Friction**: Transaction costs, brokerage, taxes (CTT/GST/stamp duty), and execution slippage are assumed benchmarks.
- **Volume vs Depth**: Traded volume does not equate to executable order book depth.
- **Disclaimer**: This research and software is for analytical demonstration only and does not constitute financial or trading advice.

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
pytest -v tests/test_core.py
```

### 3. Launch Dashboard
```bash
streamlit run streamlit/app.py
```
Open your browser at `http://localhost:8501`.

### 4. Interactive Research Walkthrough
Open **`walkthrough.ipynb`** directly on GitHub or in Jupyter/VS Code for the complete step-by-step narrative with pre-rendered interactive charts and analysis tables.

---

## Project Structure
```text
MCX GOLD/
├── walkthrough.ipynb          # Master research walkthrough notebook (all charts & analysis)
├── streamlit/                 # Dedicated interactive dashboard folder
│   ├── app.py                 # Streamlit web dashboard user interface
│   └── README.md              # Dashboard guide and feature documentation
├── scripts/                   # Individual standalone pipeline & analysis scripts
│   ├── preprocess.py          # Bhavcopy data ingestion and cleaning pipeline
│   ├── analyze_gaps.py        # Gap distribution and statistical evaluation script
│   ├── goldm_check.py         # GOLDM reference interpolation & proxy validation
│   ├── ddr_check.py           # Due Date Rate (DDR) settlement convergence analysis
│   └── backtest_simple.py     # Pairs trading backtest engine and performance logger
├── tests/
│   └── test_core.py           # Pytest test suite for core calculations and logic
├── core.py                    # Calculation and analytical core logic (no Streamlit code)
├── requirements.txt           # Project dependencies
├── data/                      # Raw daily Bhavcopy CSV files
├── out/
│   └── gold_futures_clean.csv # Cleaned, normalized dataset
├── results/                   # Analytical tables and generated visualizations
├── results_ddr/
│   └── ddr.csv                # Official MCX Due Date Rate records
├── results_bt/
│   ├── trades.csv             # Backtest trade log output
│   └── strategy_vs_gold.png   # Cumulative strategy performance chart
├── results_goldm/             # GOLDM reference analysis artifacts
└── slide_assets/              # Visual diagrams and presentation assets
```

---

## Data Source
- Daily Bhavcopy reports published by the **Multi Commodity Exchange of India (MCX)**.
- Official **MCX Contract Specifications** for GOLDM, GOLDTEN, GOLDGUINEA, and GOLDPETAL.

