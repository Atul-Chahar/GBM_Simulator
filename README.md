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

30-day walk-forward backtest over 720 hourly predictions with strict no-peeking enforcement:

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| **Coverage (95%)** | 95.28% | ~95.00% | ✅ ON TARGET (+0.28%) |
| **Average Width** | $1,225.83 | As narrow as possible | ✅ Competitive |
| **Mean Winkler Score** | 1,683.47 | Lower is better | ✅ Good |
| **Total Predictions** | 720 | 720 (30 days × 24h) | ✅ Complete |
| **Hits / Misses** | 686 / 34 | — | — |

## 🏗️ Architecture

```
GBM_simulator/
├── app.py                     # Streamlit dashboard (Part B + C)
├── backtest.py                # 30-day walk-forward backtest (Part A)
├── model/
│   ├── __init__.py            # Package exports
│   ├── data_fetcher.py        # Binance API data pipeline
│   ├── gbm_engine.py          # GBM + FIGARCH simulation engine
│   └── evaluator.py           # Coverage, Width, Winkler metrics
├── persistence/
│   ├── __init__.py            # Package exports
│   └── storage.py             # Prediction history (JSON + Google Sheets)
├── static/
│   └── styles.css             # xAI-inspired design system
├── .streamlit/
│   └── config.toml            # Streamlit server + theme config
├── backtest_results.jsonl     # Part A output (720 predictions)
├── requirements.txt           # Python dependencies
├── design.md                  # UI design system reference
├── brief.md                   # Original challenge brief
├── ARCHITECTURE.md            # Technical deep-dive (this repo)
├── CONTRIBUTING.md            # Contributor guide
└── README.md                  # This file
```

## 🧠 How the Model Works

### Core Pipeline

```
Raw Prices → Log Returns → FIGARCH Volatility → Student-t Noise → Monte Carlo → 95% CI
```

**Step-by-step:**

1. **Data Ingestion** (`data_fetcher.py`): Fetch the latest 500 closed 1-hour BTCUSDT bars from Binance's public API (`data-api.binance.vision` — no API key needed, no geo-block).

2. **Log Returns**: Transform close prices to log returns: `r_t = ln(P_t / P_{t-1})`. This makes the multiplicative price process additive and approximately stationary.

3. **FIGARCH(1,0,1)** (`gbm_engine.py`): Fit a Fractionally Integrated GARCH model to capture **long-memory volatility clustering**. Unlike standard GARCH which assumes exponential decay, FIGARCH captures the hyperbolic decay observed in Bitcoin — where volatility shocks persist for a very long time.

4. **Student-t Innovations**: Instead of assuming normally distributed returns (which would systematically underestimate extreme moves), we use a Student-t distribution. The degrees of freedom (ν ≈ 5–12) are estimated from the data, naturally adapting to BTC's fat tails.

5. **Entropy-Based Crisis Detection**: Compute rolling Shannon entropy of standardized residuals to detect chaotic/uncertain market regimes. When entropy spikes above a threshold, the model automatically widens its prediction range.

6. **Monte Carlo Simulation**: Generate 10,000 possible "next hour" paths using the fitted volatility and Student-t noise. The 95% confidence interval is read off as the 2.5th and 97.5th percentiles of the simulated terminal prices.

7. **Calibration**: A variance calibration factor (0.85) is applied to align empirical coverage with the 95% target. This compensates for model-inherent conservatism.

### Three Critical Concepts

| Concept | What It Means | How We Handle It |
|---------|---------------|------------------|
| **No Peeking** | Can't use bar N's price to predict bar N | Walk-forward: train on `[i-500, i)`, predict bar `i` |
| **Volatility Clustering** | Calm/violent hours cluster together | FIGARCH captures long-memory; entropy detects regime shifts |
| **Fat Tails** | BTC has more extreme moves than Normal predicts | Student-t distribution with estimated ν ≈ 5–12 |

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Internet connection (for Binance public API)
- No API keys needed

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
- Fetch ~1,220 hourly bars from Binance (~5 seconds)
- Run 720 walk-forward predictions (~25–40 minutes on CPU)
- Print progress every 10 bars with live coverage tracking
- Save results to `backtest_results.jsonl`
- Print final coverage, average width, and Winkler score

Expected output:
```
============================================================
  BACKTEST RESULTS
============================================================
  Total predictions : 720
  Coverage (95%)    : 95.3%
  Average Width     : $1,225.83
  Mean Winkler Score: 1,683.47
  Hits / Misses     : 686 / 34
============================================================
  ✅ Coverage is excellent (close to target 0.95)
```

### Run the Dashboard (Part B)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with:
- Live BTC price (auto-refreshes every 5 minutes)
- 95% confidence interval for the next hour
- Price chart with CI ribbon (last 50 bars)
- Three additional analytics panels (hourly returns, FIGARCH volatility, Monte Carlo distribution)
- Market statistics (24h / 7d)
- Model parameters
- Prediction history (Part C)
- Dark / Light theme toggle

## 🎨 Dashboard Design

The dashboard uses an **xAI-inspired brutalist minimalist** design system:

- **Typography**: JetBrains Mono (display/labels) + Inter (body)
- **Colors**: Monochromatic white-on-dark (`#1f2228`) with functional green accents for positive data
- **Depth**: Zero shadows — elevation through border opacity only
- **Layout**: Card-based grid with generous whitespace
- **Animations**: Staggered `fadeInUp` entrance, live pulse indicator, hover transitions
- **Themes**: Full dark/light toggle via CSS custom properties

See [`design.md`](design.md) for the full design system specification.

## 🔍 Bugs Found in Starter Notebook

1. **`pip install arch` as bare Python** (line 10): Should be `!pip install arch` (notebook magic command) or removed entirely for a Python script. As-is, it's interpreted as an expression statement.

2. **Global variable leakage**: `simulate_cyber_gbm()` references `redundancy` and `info_filter` from global scope instead of accepting them as parameters. This creates hidden dependencies and makes the function non-reentrant.

3. **`dt` parameter mismatch**: The FIGARCH model is fitted on hourly log returns, so its output variance is already per-hour. Using `dt = 1/24` (one hour as fraction of a day) incorrectly scales the variance down by 24×, producing intervals that are far too narrow. Correct value: `dt = 1.0`.

4. **Backtest window indexing**: The `backtest_confidence_intervals` function uses `iloc[:-1]` on entropy and momentum series, which can misalign indicators relative to training window boundaries.

5. **Untranslated French**: Labels like "Aujourd'hui", "Demain", "Prix Réel" are from the original French notebook and should be translated for an English submission.

## 📋 Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Streamlit** | Brief recommends it. Free hosting on Community Cloud, Python-only, fast to ship. |
| **FIGARCH over GARCH** | Long-memory volatility is empirically observed in BTC. FIGARCH captures this; standard GARCH loses it. |
| **Student-t over Normal** | BTC returns have excess kurtosis (~5–12 df). Normal underestimates tails → coverage < 95%. |
| **500-bar sliding window** | Brief specifies "last 500 bars". Prevents stale regime dominance. |
| **dt = 1.0** | FIGARCH variance is already per-hour. Using 1/24 scales it wrong — this is the critical bug fix. |
| **Calibration factor 0.85** | Empirically tuned on validation set to align coverage with 95% target. |
| **Google Sheets persistence** | Streamlit Cloud has ephemeral filesystem. Sheets provides free, persistent storage. |

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
| `dt` | 1.0 | One hour — FIGARCH already estimates per-hour variance |
| `confidence` | 0.95 | 95% CI as specified |
| `FIGARCH(p,o,q)` | (1,0,1) | From starter notebook; captures long memory |
| `dist` | Student-t | Fat tails; from starter notebook |
| `calibration` | 0.85 | Empirical tuning for 95% coverage alignment |

## 🌐 Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select the repo and `app.py` as the main file
5. (Optional) Add Google Sheets secrets under **Settings → Secrets** for Part C persistence

The app auto-sleeps after idle and wakes in ~30 seconds when visited.

## 📝 License

MIT — Built for the AlphaI × Polaris internship challenge.
