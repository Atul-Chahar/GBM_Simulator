"""
evaluator.py — Prediction Evaluation Metrics
=============================================
Implements the three key metrics from the challenge brief:

1. Coverage — Fraction of predictions that contained the actual price
2. Average Width — How wide the typical prediction interval is
3. Winkler Score — Combined accuracy + tightness (lower = better)

The Winkler score formula:
    - If actual is inside the interval: score = width
    - If actual is below the interval:  score = width + (2/α) × (low - actual)
    - If actual is above the interval:  score = width + (2/α) × (actual - high)

Where α = 0.05 for a 95% confidence interval.
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import List, Dict, Tuple


def winkler_score(
    actual: float,
    low: float,
    high: float,
    alpha: float = 0.05,
) -> float:
    """
    Compute the Winkler score for a single prediction interval.

    Parameters
    ----------
    actual : float
        The observed actual price.
    low : float
        Lower bound of the prediction interval.
    high : float
        Upper bound of the prediction interval.
    alpha : float
        Significance level (default 0.05 for 95% CI).

    Returns
    -------
    float
        Winkler score. Lower is better.
    """
    width = high - low

    if actual < low:
        # Actual below interval → penalize proportional to distance
        return width + (2 / alpha) * (low - actual)
    elif actual > high:
        # Actual above interval → penalize proportional to distance
        return width + (2 / alpha) * (actual - high)
    else:
        # Actual inside interval → score equals width (reward narrow intervals)
        return width


def coverage_hit(actual: float, low: float, high: float) -> int:
    """
    Check if the actual price fell inside the predicted interval.

    Returns
    -------
    int
        1 if actual ∈ [low, high], 0 otherwise.
    """
    return int(low <= actual <= high)


def evaluate_predictions(
    predictions: List[Dict],
) -> Dict[str, float]:
    """
    Compute aggregate evaluation metrics for a list of predictions.

    Parameters
    ----------
    predictions : list[dict]
        Each dict must contain:
            - "actual_close": float
            - "predicted_low_95": float
            - "predicted_high_95": float

    Returns
    -------
    dict
        {
            "coverage_95": float,       # Fraction of hits (target: ~0.95)
            "avg_width_95": float,      # Mean interval width
            "mean_winkler_95": float,   # Mean Winkler score (lower = better)
            "total_predictions": int,
            "total_hits": int,
            "total_misses": int,
        }
    """
    if not predictions:
        return {
            "coverage_95": 0.0,
            "avg_width_95": 0.0,
            "mean_winkler_95": 0.0,
            "total_predictions": 0,
            "total_hits": 0,
            "total_misses": 0,
        }

    hits = 0
    total_width = 0.0
    total_winkler = 0.0
    n = len(predictions)

    for pred in predictions:
        actual = pred["actual_close"]
        low = pred["predicted_low_95"]
        high = pred["predicted_high_95"]

        # Coverage
        hit = coverage_hit(actual, low, high)
        hits += hit

        # Width
        width = high - low
        total_width += width

        # Winkler
        w = winkler_score(actual, low, high)
        total_winkler += w

    return {
        "coverage_95": hits / n,
        "avg_width_95": total_width / n,
        "mean_winkler_95": total_winkler / n,
        "total_predictions": n,
        "total_hits": hits,
        "total_misses": n - hits,
    }


def format_metrics_display(metrics: Dict[str, float]) -> Dict[str, str]:
    """
    Format metrics for dashboard display.

    Returns
    -------
    dict
        Human-readable formatted strings.
    """
    return {
        "coverage": f"{metrics['coverage_95']:.1%}",
        "avg_width": f"${metrics['avg_width_95']:,.2f}",
        "winkler": f"{metrics['mean_winkler_95']:,.2f}",
        "predictions": str(metrics["total_predictions"]),
        "hits": str(metrics["total_hits"]),
        "misses": str(metrics["total_misses"]),
    }


def predictions_to_dataframe(
    predictions: List[Dict],
) -> pd.DataFrame:
    """
    Convert prediction list to a DataFrame for analysis/display.
    """
    df = pd.DataFrame(predictions)

    if "bar_timestamp" in df.columns:
        df["bar_timestamp"] = pd.to_datetime(df["bar_timestamp"])

    if all(
        col in df.columns
        for col in ["actual_close", "predicted_low_95", "predicted_high_95"]
    ):
        df["width"] = df["predicted_high_95"] - df["predicted_low_95"]
        df["hit"] = (
            (df["actual_close"] >= df["predicted_low_95"])
            & (df["actual_close"] <= df["predicted_high_95"])
        ).astype(int)
        df["winkler"] = df.apply(
            lambda r: winkler_score(
                r["actual_close"], r["predicted_low_95"], r["predicted_high_95"]
            ),
            axis=1,
        )

    return df


def compute_calibration_curve(
    predictions: List[Dict],
    confidence_levels: list = None,
) -> dict:
    """
    Compute reliability diagram (calibration curve).

    Compares expected coverage vs actual coverage across confidence levels.
    For each level, rescales the stored 95% intervals proportionally.
    """
    if confidence_levels is None:
        confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]

    df = predictions_to_dataframe(predictions)

    ref_conf = 0.95
    ref_z = stats.t.ppf(1 - (1 - ref_conf) / 2, df=max(4, len(df) - 1))

    expected = []
    actual = []

    for conf in confidence_levels:
        alpha = 1 - conf
        z = stats.t.ppf(1 - alpha / 2, df=max(4, len(df) - 1))
        scale = z / ref_z

        mid = (df["predicted_low_95"] + df["predicted_high_95"]) / 2
        width = (df["predicted_high_95"] - df["predicted_low_95"]) / 2
        adj_low = mid - width * scale
        adj_high = mid + width * scale

        hits = ((df["actual_close"] >= adj_low) & (df["actual_close"] <= adj_high)).mean()
        actual_cov = float(hits) if not pd.isna(hits) else conf

        expected.append(conf)
        actual.append(actual_cov)

    return {"expected": expected, "actual": actual}


def decompose_winkler(predictions: List[Dict]) -> dict:
    """
    Decompose Winkler score into width and penalty components.

    Returns mean width (from hits) and mean penalty (from misses).
    """
    df = predictions_to_dataframe(predictions)
    hits_df = df[df["hit"] == 1]
    misses_df = df[df["hit"] == 0]

    mean_width = float(hits_df["winkler"].mean()) if len(hits_df) > 0 else 0.0
    mean_penalty = 0.0
    if len(misses_df) > 0:
        hit_winklers = hits_df["winkler"].sum() if len(hits_df) > 0 else 0.0
        total_winklers = df["winkler"].sum()
        mean_penalty = float((total_winklers - hit_winklers) / len(misses_df))

    penalty_ratio = mean_penalty / (mean_width + mean_penalty) if (mean_width + mean_penalty) > 0 else 0.0

    return {
        "mean_width": mean_width,
        "mean_penalty": mean_penalty,
        "penalty_ratio": penalty_ratio,
        "n_hits": len(hits_df),
        "n_misses": len(misses_df),
    }


# ─── Quick self-test ─────────────────────────────────────────────
if __name__ == "__main__":
    # Test with sample predictions
    sample = [
        {"actual_close": 67650, "predicted_low_95": 67200, "predicted_high_95": 67800},
        {"actual_close": 68400, "predicted_low_95": 67500, "predicted_high_95": 68100},
        {"actual_close": 67900, "predicted_low_95": 67600, "predicted_high_95": 68200},
    ]

    metrics = evaluate_predictions(sample)
    formatted = format_metrics_display(metrics)

    print("Evaluation Results:")
    for k, v in formatted.items():
        print(f"  {k}: {v}")

    # Test Winkler for each case
    print("\nIndividual Winkler scores:")
    for p in sample:
        w = winkler_score(p["actual_close"], p["predicted_low_95"], p["predicted_high_95"])
        hit = "✅" if coverage_hit(p["actual_close"], p["predicted_low_95"], p["predicted_high_95"]) else "❌"
        print(f"  {hit} actual={p['actual_close']}, range=[{p['predicted_low_95']}, {p['predicted_high_95']}], winkler={w:.2f}")
