"""
baselines.py — Naive Baseline Models for Comparison
====================================================
ATR-based and constant-width ranges as benchmarks against
the GBM+FIGARCH model.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats


def atr_range(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
    confidence: float = 0.95,
) -> tuple:
    """
    ATR-based prediction range.

    Parameters
    ----------
    high, low, close : pd.Series
        OHLC price data (aligned index).
    period : int
        ATR lookback period (default 14).
    confidence : float
        Confidence level (default 0.95).

    Returns
    -------
    tuple : (low_bound, high_bound, half_width)
        Half-width is based on ATR × t-critical value.
    """
    tr = pd.concat([
        high - low,
        (high - close).abs(),
        (low - close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=len(close) - period)

    hw = atr.iloc[-1] * t_crit if len(close) > period else tr.mean() * t_crit
    mid = close.iloc[-1]
    return mid - hw, mid + hw, hw


def constant_width_range(
    current_price: float,
    width_pct: float = 0.01,
    confidence: float = 0.95,
) -> tuple:
    """
    Constant-width prediction range (±width_pct from current price).

    Parameters
    ----------
    current_price : float
        Latest close price.
    width_pct : float
        Width as fraction of price (default 0.01 = ±1%).
    confidence : float
        Confidence level (used for t-critical scaling).

    Returns
    -------
    tuple : (low_bound, high_bound, half_width)
    """
    alpha = 1 - confidence
    t_crit = stats.t.ppf(1 - alpha / 2, df=30)
    hw = current_price * width_pct * t_crit
    return current_price - hw, current_price + hw, hw


def compare_models(
    predictions: list,
    prices: pd.Series = None,
    high: pd.Series = None,
    low: pd.Series = None,
) -> dict:
    """
    Compare GBM+FIGARCH predictions against naive baselines.

    Parameters
    ----------
    predictions : list[dict]
        List of prediction dicts with actual_close, predicted_low_95,
        predicted_high_95.
    prices : pd.Series, optional
        Close prices for ATR computation.
    high, low : pd.Series, optional
        High/Low prices for ATR computation.

    Returns
    -------
    dict with model comparison table
    """
    from model.evaluator import evaluate_predictions, coverage_hit, winkler_score

    result = {"gbm": evaluate_predictions(predictions)}

    if prices is not None and high is not None and low is not None:
        atr_preds = []
        for i in range(len(predictions)):
            try:
                h = high.iloc[:i + 1]
                l = low.iloc[:i + 1]
                c = prices.iloc[:i + 1]
                al, ah, _ = atr_range(h, l, c)
                actual = predictions[i]["actual_close"]
                hit = coverage_hit(actual, al, ah)
                w = winkler_score(actual, al, ah)
                width = ah - al
                atr_preds.append({
                    "actual_close": actual,
                    "predicted_low_95": al,
                    "predicted_high_95": ah,
                    "coverage_95": hit,
                    "width_95": width,
                    "winkler_95": w,
                    "hit": hit,
                })
            except Exception:
                pass
        if atr_preds:
            result["atr"] = evaluate_predictions(atr_preds)

    const_preds = []
    for p in predictions:
        try:
            cp = p["current_price"]
            cl, ch, _ = constant_width_range(cp, 0.01)
            actual = p["actual_close"]
            hit = coverage_hit(actual, cl, ch)
            w = winkler_score(actual, cl, ch)
            width = ch - cl
            const_preds.append({
                "actual_close": actual,
                "predicted_low_95": cl,
                "predicted_high_95": ch,
                "coverage_95": hit,
                "width_95": width,
                "winkler_95": w,
                "hit": hit,
            })
        except Exception:
            pass
    if const_preds:
        result["constant"] = evaluate_predictions(const_preds)

    return result