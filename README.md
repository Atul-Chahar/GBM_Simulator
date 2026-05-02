# ₿ BTC Next-Hour Forecast — AlphaI × Polaris Challenge

> Predict Bitcoin's next-hour price range using a Geometric Brownian Motion simulator enhanced with FIGARCH volatility modeling and Student-t fat-tail innovations.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🎯 Challenge Overview

Every hour, a new candle closes on Bitcoin's chart. This project predicts the **95% confidence interval** where BTC will land one hour from now — not an exact price, but a *range*.

The best forecaster is one that is:
1. **Accurate** — right ~95% of the time
2. **Tight** — keeps the range as narrow as possible

## 📊 Backtest Results (Part A)

| Metric | Value | Target |
|--------|-------|--------|
| **Coverage (95%)** | *Run backtest to fill* | ~0.95 |
| **Average Width** | *Run backtest to fill* | As narrow as possible |
| **Mean Winkler Score** | *Run backtest to fill* | Lower is better |
| **Total Predictions** | ~720 | 720 (30 days × 24h) |

## 🏗️ Architecture

```
GBM_simulator/
├── app.py                  # Streamlit dashboard (Part B + C)
├── backtest.py             # 30-day walk-forward backtest (Part A)
├── model/
│   ├── data_fetcher.py     # Binance API data pipeline
│   ├── gbm_engine.py       # GBM + FIGARCH simulation engine
│   └── evaluator.py        # Coverage, Width, Winkler metrics
├── persistence/
│   └── storage.py          # Prediction history persistence
├── backtest_results.jsonl  # Part A output
├── requirements.txt        # Python dependencies
├── .streamlit/config.toml  # Dashboard theme config
└── README.md               # This file
```

## 🧠 How the Model Works

### Core Pipeline

```
Raw Prices → Log Returns → FIGARCH Volatility → Monte Carlo Simulation → 95% CI
```

1. **Data Ingestion**: Fetch the latest 500 closed 1-hour BTCUSDT bars from Binance's public API (`data-api.binance.vision` — no API key needed)

2. **Log Returns**: Transform close prices to log returns: `ln(P_t / P_{t-1})`

3. **FIGARCH(1,0,1)**: Fit a Fractionally Integrated GARCH model to capture **long-memory volatility clustering**. Unlike standard GARCH which assumes exponential decay, FIGARCH captures the hyperbolic decay observed in Bitcoin — where volatility shocks persist for a very long time.

4. **Student-t Innovations**: Instead of assuming normally distributed returns (which would systematically underestimate extreme moves), we use a Student-t distribution to model the **fat tails** that Bitcoin exhibits. The degrees of freedom (ν) are estimated from the data.

5. **Entropy-Based Crisis Detection**: We compute rolling Shannon entropy of standardized residuals to detect chaotic/uncertain market regimes. When entropy spikes, the model automatically widens its prediction range.

6. **Monte Carlo Simulation**: Generate 10,000 possible "next hour" paths using the fitted volatility and Student-t noise. The 95% range is the 2.5th and 97.5th percentiles of these simulated terminal prices.

### Three Critical Concepts

| Concept | What It Means | How We Handle It |
|---------|---------------|------------------|
| **No Peeking** | Can't use bar N's price to predict bar N | Walk-forward: train on `[i-500, i)`, predict `i` |
| **Volatility Clustering** | Calm/violent hours cluster together | FIGARCH captures long-memory; entropy detects regime shifts |
| **Fat Tails** | BTC has more extreme moves than Normal predicts | Student-t distribution with estimated ν ≈ 5-12 |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Internet connection (for Binance API)

### Setup

```bash
# Clone the repository
git clone <repo-url>
cd GBM_simulator

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Backtest (Part A)

```bash
python backtest.py
```

This will:
- Fetch ~1220 hourly bars from Binance
- Run 720 walk-forward predictions (takes ~30-60 minutes)
- Save results to `backtest_results.jsonl`
- Print coverage, average width, and Winkler score

### Run the Dashboard (Part B)

```bash
streamlit run app.py
```

This will:
- Open a browser at `http://localhost:8501`
- Fetch the latest 500 bars from Binance
- Run the model and display the prediction
- Show the backtest metrics from Part A
- Display prediction history (Part C)

## 🔍 Bugs Found in Starter Notebook

1. **`pip install arch` as bare Python** (line 10): Should be `!pip install arch` (notebook magic command) or removed entirely for a Python script. As-is, it's interpreted as an expression statement, not a shell command.

2. **Global variable leakage**: `simulate_cyber_gbm()` references `redundancy` and `info_filter` from global scope instead of accepting them as parameters. This creates hidden dependencies and makes the function non-reentrant.

3. **`dt` parameter mismatch**: When adapting from daily to hourly data, `dt = 1` (one day) must become `dt = 1/24` (one hour as a fraction of a day). The drift and volatility terms scale with `dt`, so using the wrong value produces wildly incorrect intervals.

4. **Backtest window indexing**: The `backtest_confidence_intervals` function uses `iloc[:-1]` on entropy and momentum series, which can misalign the indicators relative to the training window boundaries.

5. **Untranslated French**: Labels like "Aujourd'hui", "Demain", "Prix Réel" are from the original French notebook and should be translated for an English submission.

## 📋 Design Decisions

### Why Streamlit?
The brief recommends it as the "easiest route" — free hosting on Streamlit Community Cloud, simple Python-only development, and built-in interactivity. For a time-constrained challenge, shipping speed matters more than framework sophistication.

### Why Google Sheets for Persistence?
Streamlit Community Cloud has an **ephemeral filesystem** — local files are lost on reboot. Google Sheets provides free, persistent, human-readable storage without requiring a database server. The trade-off is write latency (~200ms per row), which is acceptable for hourly prediction logging.

### Why Not a Normal Distribution?
Bitcoin returns exhibit excess kurtosis (fat tails) — extreme moves happen ~3-5x more often than a Normal distribution predicts. Using Normal would systematically underestimate tail risk, leading to intervals that are too narrow and coverage well below 95%. The Student-t distribution with estimated degrees of freedom handles this gracefully.

### Walk-Forward vs. Expanding Window
We use a **fixed sliding window of 500 bars** (~21 days) rather than an expanding window. This prevents the model from being dominated by stale historical regimes and keeps it responsive to current market conditions.

## 📄 Output Format

### `backtest_results.jsonl`

Each line is a JSON object:
```json
{
  "bar_timestamp": "2026-04-03T14:00:00+00:00",
  "current_price": 78432.50,
  "predicted_low_95": 78200.00,
  "predicted_high_95": 78800.00,
  "actual_close": 78650.00,
  "coverage_95": 1,
  "width_95": 600.00,
  "winkler_95": 600.00
}
```

## 🔧 Configuration

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_sims` | 10,000 | Brief specifies "10,000 possible next hours" |
| `train_window` | 500 bars | Brief says "last 500 bars" |
| `dt` | 1/24 | One hour as fraction of a day |
| `confidence` | 0.95 | 95% CI as specified |
| `FIGARCH(p,o,q)` | (1,0,1) | From starter notebook; captures long memory |
| `dist` | Student-t | Fat tails; from starter notebook |

## 📝 License

MIT — Built for the AlphaI × Polaris internship challenge.
