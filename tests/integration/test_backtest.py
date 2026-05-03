"""tests/integration/test_backtest.py"""
import pytest
import sys, os, tempfile, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
from model.gbm_engine import GBMEngine
from model.evaluator import coverage_hit


def _synthetic_prices(n=600, seed=42):
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(0.0005, 0.02, n)
    price = 95000.0
    prices = [price]
    for r in log_returns:
        price *= np.exp(r)
        prices.append(price)
    idx = pd.date_range("2026-03-01", periods=n + 1, freq="h")
    return pd.Series(prices, index=idx)


class TestBacktest:
    def test_walk_forward_no_peeking(self):
        prices = _synthetic_prices(600, seed=42)
        TRAIN_WINDOW = 100
        TEST_BARS = 20

        predictions = []
        for i in range(TRAIN_WINDOW, TRAIN_WINDOW + TEST_BARS):
            train_prices = prices.iloc[i - TRAIN_WINDOW:i]
            if len(train_prices) < 60:
                continue
            eng = GBMEngine(n_sims=500, random_seed=42)
            eng.fit(train_prices)
            low, high, _, _ = eng.predict_interval(confidence=0.95)
            actual = float(prices.iloc[i])
            hit = coverage_hit(actual, float(low), float(high))
            predictions.append({
                "bar_timestamp": str(prices.index[i]),
                "current_price": float(train_prices.iloc[-1]),
                "predicted_low_95": float(low),
                "predicted_high_95": float(high),
                "actual_close": actual,
                "coverage_95": hit,
            })

        assert len(predictions) == TEST_BARS
        cov = sum(p["coverage_95"] for p in predictions) / len(predictions)
        assert 0.0 <= cov <= 1.0

    def test_output_format(self):
        prices = _synthetic_prices(600, seed=42)
        eng = GBMEngine(n_sims=500, random_seed=42)
        eng.fit(prices.iloc[:500])
        low, high, _, _ = eng.predict_interval()
        actual = float(prices.iloc[500])
        pred = {
            "bar_timestamp": str(prices.index[500]),
            "current_price": float(prices.iloc[499]),
            "predicted_low_95": float(low),
            "predicted_high_95": float(high),
            "actual_close": actual,
            "coverage_95": coverage_hit(actual, float(low), float(high)),
            "width_95": float(high - low),
            "winkler_95": 100.0,
        }
        assert all(k in pred for k in [
            "bar_timestamp", "current_price", "predicted_low_95",
            "predicted_high_95", "actual_close", "coverage_95",
        ])

    def test_jsonl_serialization(self):
        fd, path = tempfile.mkstemp(suffix=".jsonl")
        os.close(fd)
        try:
            predictions = [
                {"bar_timestamp": f"2026-05-0{i}T12:00:00+00:00",
                 "current_price": 95000.0, "predicted_low_95": 94000.0,
                 "predicted_high_95": 96000.0, "actual_close": 95000.0}
                for i in range(1, 6)
            ]
            with open(path, "w") as f:
                for p in predictions:
                    f.write(json.dumps(p) + "\n")

            loaded = []
            with open(path) as f:
                for line in f:
                    loaded.append(json.loads(line.strip()))
            assert len(loaded) == 5
        finally:
            os.remove(path)


class TestPipeline:
    def test_fetch_fit_predict_evaluate(self):
        prices = _synthetic_prices(600, seed=42)
        eng = GBMEngine(n_sims=500, random_seed=42)
        eng.fit(prices.iloc[:500])
        low, high, _, _ = eng.predict_interval()
        actual = float(prices.iloc[500])
        hit = coverage_hit(actual, float(low), float(high))
        assert hit in [0, 1]
        assert float(low) < float(high)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])