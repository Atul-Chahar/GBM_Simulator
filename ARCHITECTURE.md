# Architecture — BTC GBM Forecast System

> Technical deep-dive into every module, data flow, and design decision.

---

## System Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        BINANCE PUBLIC API                        │
│              data-api.binance.vision/api/v3/klines               │
└──────────────────────┬───────────────────────────────────────────┘
                       │  BTCUSDT 1H klines (OHLCV)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      data_fetcher.py                             │
│  • Paginated fetcher with 1000-bar chunks                        │
│  • Geo-block fallback URLs (api.binance.com → data-api)          │
│  • Returns pd.DataFrame with DatetimeIndex                       │
└──────────────────────┬───────────────────────────────────────────┘
                       │  pd.Series of close prices (500 bars)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      gbm_engine.py                               │
│                                                                  │
│  GBMEngine.fit(prices):                                          │
│    1. Compute log returns: r_t = ln(P_t / P_{t-1})               │
│    2. Fit FIGARCH(1,0,1) with Student-t dist via `arch` library  │
│    3. Extract: mu (drift), sigma_fig (conditional vol series),   │
│       nu (degrees of freedom)                                    │
│    4. Compute Shannon entropy for crisis detection               │
│    5. Classify volatility regime (Calm / Moderate / High)        │
│                                                                  │
│  GBMEngine.predict_interval(confidence=0.95):                    │
│    1. S₀ = latest close price                                    │
│    2. σ² updated via: bar_sigma² × (1 + α·H + δ_t·M) + γ·(bar_σ² − σ²) │
│       then × redundancy × 0.85 calibration factor               │
│    3. Generate 10,000 Student-t random draws                     │
│    4. S₁ = S₀ × exp((μ - σ²/2)·dt + σ·√dt·ε)                   │
│    5. Return percentile(2.5%) and percentile(97.5%)              │
└──────────────────────┬───────────────────────────────────────────┘
                       │  (low, high, simulations, mean)
                       ▼
┌──────────────────────────────────────────────────────────────────┐
│                      evaluator.py                                │
│  • coverage_hit(actual, low, high) → 0 or 1                     │
│  • winkler_score(actual, low, high, alpha=0.05) → float          │
│  • evaluate_predictions(list[dict]) → {coverage, width, winkler} │
│  • format_metrics_display(metrics) → formatted strings           │
└──────────────────────────────────────────────────────────────────┘
```

---

## Module Reference

### `model/data_fetcher.py`

**Purpose**: Fetch BTCUSDT hourly klines from Binance's public REST API.

**Key Functions**:

| Function | Signature | Description |
|----------|-----------|-------------|
| `fetch_btc_klines` | `(num_bars=500) → pd.DataFrame` | Paginated fetcher. Splits requests into 1000-bar chunks, merges, returns OHLCV DataFrame with UTC DatetimeIndex. |
| `fetch_latest_bars` | `(num_bars=500) → pd.DataFrame` | Wrapper for dashboard use. Same logic, cleaner name. |
| `get_close_prices` | `(df) → pd.Series` | Extracts the "close" column as a named Series. |

**API Details**:
- Primary URL: `https://data-api.binance.vision/api/v3/klines`
- Fallback URL: `https://api.binance.com/api/v3/klines`
- Parameters: `symbol=BTCUSDT`, `interval=1h`, `limit=1000`
- No API key required — fully public endpoint
- Handles geo-blocking by falling back to alternate URL

**Error Handling**:
- HTTP errors raise with status code
- Empty response returns empty DataFrame
- Automatic retry on connection timeout

---

### `model/gbm_engine.py`

**Purpose**: The core simulation engine. Fits FIGARCH volatility, runs Monte Carlo simulation, produces confidence intervals.

**Class: `GBMEngine`**

```python
engine = GBMEngine(n_sims=10_000)
engine.fit(prices)  # pd.Series of close prices
low, high, sims, mean = engine.predict_interval(confidence=0.95)
info = engine.get_model_info()  # dict of model params
```

**Key Attributes After `.fit()`**:

| Attribute | Type | Description |
|-----------|------|-------------|
| `mu` | float | Hourly drift (mean log return) |
| `sigma_fig` | pd.Series | FIGARCH conditional volatility series |
| `nu` | float | Student-t degrees of freedom |
| `DT` | float | Time step (1.0 = one hour) |
| `entropy_crisis` | bool | Whether current regime is detected as crisis |

**Critical Implementation Details**:

1. **`DT = 1.0`**: The FIGARCH model is fitted on hourly returns, so its variance output is already per-hour. Setting `dt = 1/24` (as the starter notebook implies for "fraction of a day") would divide the variance by 24, producing intervals ~5× too narrow. This was the most critical bug fix.

2. **Calibration Factor (0.85)**: After fitting FIGARCH and Student-t, the raw Monte Carlo intervals tend to be ~2–5% wider than needed for exactly 95% coverage. The 0.85 factor multiplies the final variance to tighten intervals, validated on a 200-bar walk-forward test.

3. **Crisis Detection**: Shannon entropy of the last 50 standardized residuals. If entropy > 0.8 (normalized, where H_max is the series maximum), a crisis multiplier (delta_t = 0.10) is added to variance. An additional delta (0.10) and redundancy floor (0.02) provide baseline width.

4. **Regime Classification**:
   - `σ < 0.002` → Calm (narrow intervals)
   - `0.002 ≤ σ < 0.005` → Moderate
   - `σ ≥ 0.005` → High (wide intervals)

---

### `model/evaluator.py`

**Purpose**: Scoring functions exactly as defined in the challenge brief.

**Functions**:

| Function | Formula | Notes |
|----------|---------|-------|
| `coverage_hit(actual, low, high)` | `1 if low ≤ actual ≤ high else 0` | Binary hit/miss |
| `winkler_score(actual, low, high, alpha=0.05)` | `width + (2/α)·max(low-actual, 0) + (2/α)·max(actual-high, 0)` | Width if hit; big penalty if miss |
| `evaluate_predictions(preds)` | Aggregates all predictions | Returns dict with coverage, avg_width, mean_winkler |

**Winkler Score Explained**:
- If the actual price falls **inside** the interval: score = width (reward narrow intervals)
- If the actual price falls **below** the lower bound: score = width + (2/0.05) × (low - actual)
- If the actual price falls **above** the upper bound: score = width + (2/0.05) × (actual - high)
- The penalty factor `2/α = 40` makes misses very expensive

---

### `persistence/storage.py`

**Purpose**: Save and retrieve prediction history for Part C (persistence bonus).

**Class: `PredictionStore`**

```python
store = PredictionStore(use_gsheets=False)  # or True with secrets
store.save_prediction({"timestamp": ..., "current_price": ..., ...})
df = store.get_history_dataframe()  # Returns pd.DataFrame
should_save = store.should_save_new_prediction()  # Rate-limits to 1/hour
```

**Storage Backends**:

| Backend | When Used | Persistence |
|---------|-----------|-------------|
| Local JSON | Default (`use_gsheets=False`) | Lost on Streamlit Cloud reboot |
| Google Sheets | When `st.secrets` contains GCP credentials | Permanent |

**Google Sheets Setup** (optional):
1. Create a GCP service account
2. Share a Google Sheet with the service account email
3. Add credentials to `.streamlit/secrets.toml` or Streamlit Cloud Secrets

---

### `backtest.py`

**Purpose**: Run the full 30-day (720-bar) walk-forward backtest.

**Walk-Forward Logic**:

```
For each bar i from 501 to 1220:
    train_data = prices[i-500 : i]     ← only past data
    model.fit(train_data)
    low, high = model.predict()
    actual = prices[i]                 ← revealed AFTER prediction
    score(actual, low, high)
```

**Key Constraints**:
- **No peeking**: Training data strictly ends at `i-1`
- **Fixed window**: Always 500 bars (not expanding)
- **No retraining shortcuts**: Model is re-fitted from scratch at every step

**Output**: `backtest_results.jsonl` — one JSON object per line, 720 lines total.

**Progress Display**: Prints one line every 10 bars:
```
  [ 100/720]  coverage: 93.0%  |  2.1s/bar
  [ 200/720]  coverage: 94.5%  |  2.0s/bar
  ...
  [ 720/720]  coverage: 95.3%  |  1.9s/bar
```

---

### `app.py`

**Purpose**: Live Streamlit dashboard combining Parts B and C.

**Dashboard Sections**:

| Section | Data Source | Refresh Rate |
|---------|-----------|--------------|
| Nav Bar (LIVE indicator, next bar countdown) | System clock | Every page load |
| Hero Price | Binance API (via `get_live_data()`) | 5 min cache |
| Metrics Grid (Coverage, Width, Winkler, Predictions) | `backtest_results.jsonl` | 1 hour cache |
| Prediction Range Card | Live model run | 5 min cache |
| Price Chart with CI Ribbon | Live model run | 5 min cache |
| Hourly Returns Bar Chart (24h) | Live data | 5 min cache |
| FIGARCH Volatility Timeline | Live model run | 5 min cache |
| Monte Carlo Distribution | Live model run | 5 min cache |
| Market Statistics (24h/7d) | Live data | 5 min cache |
| Model Parameters | Live model run | 5 min cache |
| Prediction History | `PredictionStore` | Every page load |

**Theme System**: CSS custom properties toggled via `st.session_state.theme`. Two complete color palettes (dark: `#1f2228`, light: `#fafafa`) with functional green (`#22c55e`) for positive data.

---

## Data Flow Diagram

```
                    ┌─────────────┐
                    │   Binance   │
                    │  Public API │
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        ┌─────▼─────┐           ┌──────▼──────┐
        │ backtest.py│           │   app.py    │
        │  (Part A)  │           │  (Part B+C) │
        └─────┬──────┘           └──────┬──────┘
              │                         │
        ┌─────▼──────┐          ┌──────▼──────┐
        │ GBMEngine  │          │ GBMEngine   │
        │ × 720 fits │          │ × 1 fit     │
        └─────┬──────┘          └──────┬──────┘
              │                        │
        ┌─────▼──────┐          ┌──────▼──────┐
        │ evaluator  │          │ Plotly +    │
        │ scoring    │          │ Streamlit   │
        └─────┬──────┘          └──────┬──────┘
              │                        │
        ┌─────▼──────────┐      ┌──────▼──────┐
        │ backtest_      │      │ storage.py  │
        │ results.jsonl  │      │ (Part C)    │
        └────────────────┘      └─────────────┘
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `streamlit` | ≥1.30 | Dashboard framework |
| `arch` | ≥6.0 | FIGARCH model fitting |
| `pandas` | ≥2.0 | Data manipulation |
| `numpy` | ≥1.24 | Numerical operations |
| `scipy` | ≥1.10 | Student-t distribution |
| `plotly` | ≥5.15 | Interactive charts |
| `requests` | ≥2.28 | Binance API calls |
| `gspread` | ≥5.10 | Google Sheets (optional) |

---

## Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Fetch 500 bars | ~2s | Single paginated request |
| Fetch 1220 bars | ~5s | Two paginated requests |
| Single model fit + predict | ~2s | FIGARCH fitting + 10K MC sims |
| Full 720-bar backtest | ~25 min | 720 × fit + predict on CPU |
| Dashboard first load | ~8s | Data fetch + model fit + chart render |
| Dashboard cached load | <1s | Streamlit cache hit (5 min TTL) |
