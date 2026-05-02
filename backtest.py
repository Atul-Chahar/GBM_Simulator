"""
backtest.py — 30-Day Walk-Forward Backtest (Part A)
====================================================
Runs a strict walk-forward backtest over the last 30 days of
BTCUSDT 1-hour bars (~720 predictions).

For EACH bar:
    1. Use ONLY data up to that bar (no peeking ahead)
    2. Fit the GBM model on the preceding 500 bars
    3. Predict the 95% confidence interval for the next bar
    4. Reveal the actual next-bar price
    5. Score the prediction (coverage + Winkler)

Output: backtest_results.jsonl (one JSON object per line)

Usage:
    python backtest.py
"""

import json
import time
import sys
import numpy as np
import pandas as pd
from datetime import datetime


from model.data_fetcher import fetch_btc_klines, get_close_prices
from model.gbm_engine import GBMEngine
from model.evaluator import (
    evaluate_predictions,
    winkler_score,
    coverage_hit,
    format_metrics_display,
)

# ── Configuration ────────────────────────────────────────────────
TRAIN_WINDOW = 500      # Number of bars for model training
TEST_BARS = 720          # ~30 days × 24 hours
N_SIMS = 10_000          # Monte Carlo simulations per prediction
OUTPUT_FILE = "backtest_results.jsonl"


def run_backtest(
    train_window: int = TRAIN_WINDOW,
    test_bars: int = TEST_BARS,
    n_sims: int = N_SIMS,
    output_file: str = OUTPUT_FILE,
) -> list:
    """
    Run the full 30-day walk-forward backtest.

    Parameters
    ----------
    train_window : int
        Number of bars to use for training at each step.
    test_bars : int
        Number of bars to test (predict).
    n_sims : int
        Number of Monte Carlo simulations per prediction.
    output_file : str
        Path to save JSONL output.

    Returns
    -------
    list[dict]
        List of prediction results.
    """
    total_bars_needed = train_window + test_bars + 1  # +1 for the actual
    print(f"{'='*60}")
    print(f"  BTC GBM Walk-Forward Backtest")
    print(f"{'='*60}")
    print(f"  Training window : {train_window} bars")
    print(f"  Test bars       : {test_bars} bars")
    print(f"  Monte Carlo sims: {n_sims:,}")
    print(f"  Total bars needed: {total_bars_needed}")
    print(f"{'='*60}\n")

    # ── Fetch data ───────────────────────────────────────────────
    print("📡 Fetching BTCUSDT 1h data from Binance...")
    start_time = time.time()
    df = fetch_btc_klines(num_bars=total_bars_needed)
    prices = get_close_prices(df)
    elapsed = time.time() - start_time
    print(f"   ✓ Fetched {len(prices)} bars in {elapsed:.1f}s")
    print(f"   Date range: {prices.index[0]} → {prices.index[-1]}")
    print(f"   Price range: ${prices.min():,.2f} → ${prices.max():,.2f}\n")

    if len(prices) < train_window + 2:
        raise ValueError(
            f"Not enough data. Got {len(prices)} bars, "
            f"need at least {train_window + 2}."
        )

    # ── Determine test range ─────────────────────────────────────
    # We want to test the last `test_bars` predictions.
    # At position i, we train on prices[i-train_window:i],
    # predict for prices.iloc[i], and compare.
    max_start = len(prices) - 1  # last index with an actual to compare
    actual_test_bars = min(test_bars, max_start - train_window)

    start_idx = max_start - actual_test_bars
    print(f"🔄 Running {actual_test_bars} predictions...")
    print(f"   Test period: {prices.index[start_idx]} → {prices.index[max_start]}\n")

    # ── Walk-forward loop ────────────────────────────────────────
    predictions = []
    engine = GBMEngine(n_sims=n_sims)
    running_hits = 0
    total_iters = max_start - start_idx
    t_start = time.time()

    for idx_count, i in enumerate(range(start_idx, max_start)):
        # STRICT NO-PEEKING: Only use data up to bar i (exclusive)
        # We predict what bar i's close price will be.
        # Training data: bars [i - train_window, i) — i.e., up to i-1
        train_end = i
        train_start = max(0, i - train_window)
        train_prices = prices.iloc[train_start:train_end]

        if len(train_prices) < 60:  # Minimum for rolling window
            continue

        # Fit model on training data
        try:
            engine.fit(train_prices)
        except Exception as e:
            # If model fails to fit, use a wide fallback interval
            current_price = train_prices.iloc[-1]
            fallback_width = current_price * 0.02  # 2% fallback
            hit = coverage_hit(
                float(prices.iloc[i]),
                float(current_price - fallback_width),
                float(current_price + fallback_width),
            )
            running_hits += hit
            predictions.append({
                "bar_timestamp": str(prices.index[i]),
                "current_price": float(current_price),
                "predicted_low_95": float(current_price - fallback_width),
                "predicted_high_95": float(current_price + fallback_width),
                "actual_close": float(prices.iloc[i]),
                "coverage_95": hit,
                "width_95": float(2 * fallback_width),
                "winkler_95": winkler_score(
                    float(prices.iloc[i]),
                    float(current_price - fallback_width),
                    float(current_price + fallback_width),
                ),
                "model_error": str(e),
            })
            if len(predictions) % 10 == 0 or len(predictions) == total_iters:
                elapsed = time.time() - t_start
                rate = elapsed / len(predictions) if predictions else 0
                cov = running_hits / len(predictions) if predictions else 0
                print(f"  [{len(predictions):>4}/{total_iters}]  coverage: {cov:.1%}  |  {rate:.1f}s/bar", flush=True)
            continue

        # Predict next bar
        try:
            low, high, sims, mean_p = engine.predict_interval(confidence=0.95)
        except Exception as e:
            current_price = train_prices.iloc[-1]
            fallback_width = current_price * 0.02
            hit = coverage_hit(
                float(prices.iloc[i]),
                float(current_price - fallback_width),
                float(current_price + fallback_width),
            )
            running_hits += hit
            predictions.append({
                "bar_timestamp": str(prices.index[i]),
                "current_price": float(current_price),
                "predicted_low_95": float(current_price - fallback_width),
                "predicted_high_95": float(current_price + fallback_width),
                "actual_close": float(prices.iloc[i]),
                "coverage_95": hit,
                "width_95": float(2 * fallback_width),
                "winkler_95": winkler_score(
                    float(prices.iloc[i]),
                    float(current_price - fallback_width),
                    float(current_price + fallback_width),
                ),
                "model_error": str(e),
            })
            if len(predictions) % 10 == 0 or len(predictions) == total_iters:
                elapsed = time.time() - t_start
                rate = elapsed / len(predictions) if predictions else 0
                cov = running_hits / len(predictions) if predictions else 0
                print(f"  [{len(predictions):>4}/{total_iters}]  coverage: {cov:.1%}  |  {rate:.1f}s/bar", flush=True)
            continue

        actual = float(prices.iloc[i])
        hit = coverage_hit(actual, float(low), float(high))
        running_hits += hit
        width = float(high - low)
        winkler = winkler_score(actual, float(low), float(high))

        predictions.append({
            "bar_timestamp": str(prices.index[i]),
            "current_price": float(train_prices.iloc[-1]),
            "predicted_low_95": float(low),
            "predicted_high_95": float(high),
            "actual_close": actual,
            "coverage_95": hit,
            "width_95": width,
            "winkler_95": winkler,
        })

        # Update progress bar with live coverage
        if len(predictions) % 10 == 0 or len(predictions) == total_iters:
            elapsed = time.time() - t_start
            rate = elapsed / len(predictions) if predictions else 0
            cov = running_hits / len(predictions) if predictions else 0
            print(f"  [{len(predictions):>4}/{total_iters}]  coverage: {cov:.1%}  |  {rate:.1f}s/bar", flush=True)

    # ── Save results ─────────────────────────────────────────────
    print(f"\n💾 Saving {len(predictions)} predictions to {output_file}...")
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    # ── Print summary ────────────────────────────────────────────
    metrics = evaluate_predictions(predictions)
    formatted = format_metrics_display(metrics)

    print(f"\n{'='*60}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*60}")
    print(f"  Total predictions : {formatted['predictions']}")
    print(f"  Coverage (95%)    : {formatted['coverage']}")
    print(f"  Average Width     : {formatted['avg_width']}")
    print(f"  Mean Winkler Score: {formatted['winkler']}")
    print(f"  Hits / Misses     : {formatted['hits']} / {formatted['misses']}")
    print(f"{'='*60}\n")

    # Coverage analysis
    cov = metrics["coverage_95"]
    if 0.93 <= cov <= 0.97:
        print("  ✅ Coverage is excellent (close to target 0.95)")
    elif cov > 0.97:
        print("  ⚠️  Coverage is too high — ranges may be too wide (conservative)")
    elif cov < 0.90:
        print("  ❌ Coverage is too low — model is overconfident")
    else:
        print("  🟡 Coverage is acceptable but could be improved")

    print(f"\n  📄 Results saved to: {output_file}")
    return predictions


def load_backtest_results(filepath: str = OUTPUT_FILE) -> list:
    """
    Load backtest results from JSONL file.

    Parameters
    ----------
    filepath : str
        Path to the JSONL file.

    Returns
    -------
    list[dict]
        List of prediction dictionaries.
    """
    predictions = []
    try:
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    predictions.append(json.loads(line))
    except FileNotFoundError:
        print(f"⚠️  Backtest results file not found: {filepath}")
        print("   Run `python backtest.py` first to generate it.")

    return predictions


# ─── Entry point ─────────────────────────────────────────────────
if __name__ == "__main__":
    run_backtest()
