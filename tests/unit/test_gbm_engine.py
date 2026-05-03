"""tests/unit/test_gbm_engine.py"""
import pytest
import numpy as np
import pandas as pd
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from model.gbm_engine import GBMEngine


def _synthetic_prices(n: int = 600, seed: int = 42) -> pd.Series:
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(0.0005, 0.02, n)
    price = 95000.0
    prices = [price]
    for r in log_returns:
        price *= np.exp(r)
        prices.append(price)
    idx = pd.date_range("2026-03-01", periods=n + 1, freq="h")
    return pd.Series(prices, index=idx)


class TestGBMEngineInit:
    def test_default_params(self):
        eng = GBMEngine()
        assert eng.n_sims == 10000
        assert eng.entropy_window == 60
        assert eng.random_seed == 42
        assert eng._fitted is False

    def test_custom_params(self):
        eng = GBMEngine(n_sims=500, entropy_window=100, random_seed=7)
        assert eng.n_sims == 500
        assert eng.entropy_window == 100
        assert eng.random_seed == 7
        assert eng.rng.randint(0, 1000000) == eng.rng.randint(0, 1000000)


class TestGBMEngineFit:
    def test_fit_sets_attributes(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        assert eng._fitted is True
        assert eng.mu is not None
        assert eng.S0 is not None
        assert eng.nu is not None
        assert eng.sigma_fig is not None
        assert eng.H_series is not None
        assert eng.M_series is not None
        assert eng.bar_sigma2 is not None
        assert eng.redundancy is not None

    def test_fit_sigma_shape(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        assert len(eng.sigma_fig) > 0
        assert eng.sigma_fig.iloc[-1] >= 0

    def test_fit_nu_minimum(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        assert eng.nu >= 4


class TestGBMEnginePredict:
    def test_not_fitted_raises(self):
        eng = GBMEngine(n_sims=100, random_seed=42)
        with pytest.raises(RuntimeError):
            eng.predict_interval()

    def test_returns_correct_types(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        low, high, sims, mean_p = eng.predict_interval()
        assert isinstance(low, float)
        assert isinstance(high, float)
        assert isinstance(sims, np.ndarray)
        assert isinstance(mean_p, float)
        assert low < high

    def test_low_less_than_high(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        for conf in [0.90, 0.95, 0.99]:
            low, high, _, _ = eng.predict_interval(confidence=conf)
            assert low < high


class TestGBMEngineReproducibility:
    def test_same_seed_same_results(self):
        prices = _synthetic_prices(600)
        eng1 = GBMEngine(n_sims=1000, random_seed=42)
        eng1.fit(prices)
        low1, high1, _, _ = eng1.predict_interval()

        eng2 = GBMEngine(n_sims=1000, random_seed=42)
        eng2.fit(prices)
        low2, high2, _, _ = eng2.predict_interval()

        assert abs(low1 - low2) < 1e-6
        assert abs(high1 - high2) < 1e-6

    def test_different_seeds_different_results(self):
        prices = _synthetic_prices(600)
        eng1 = GBMEngine(n_sims=1000, random_seed=42)
        eng1.fit(prices)
        low1, high1, _, _ = eng1.predict_interval()

        eng2 = GBMEngine(n_sims=1000, random_seed=99)
        eng2.fit(prices)
        low2, high2, _, _ = eng2.predict_interval()

        assert low1 != low2 or high1 != high2


class TestGBMEngineVolatilityRegime:
    def test_returns_string(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        regime = eng.get_volatility_regime()
        assert isinstance(regime, str)
        assert regime in ["🟢 Calm", "🟡 Moderate", "🔴 High Volatility", "Unknown"]

    def test_not_fitted_returns_unknown(self):
        eng = GBMEngine()
        assert eng.get_volatility_regime() == "Unknown"


class TestGBMEngineModelInfo:
    def test_returns_expected_fields(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        info = eng.get_model_info()
        assert "current_price" in info
        assert "hourly_drift" in info
        assert "nu_degrees_freedom" in info
        assert "latest_sigma" in info
        assert "mean_sigma2" in info
        assert "volatility_regime" in info
        assert "model_type" in info
        assert "model_label" in info

    def test_model_type_set(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        assert eng._model_type in ["FIGARCH", "GARCH"]


class TestGBMEngineCalibrationFactor:
    def test_within_bounds(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        assert 0.5 <= eng._cal_factor <= 1.5


class TestGBMEngineRedundancy:
    def test_no_inf_overflow(self):
        prices = _synthetic_prices(600)
        eng = GBMEngine(n_sims=1000, random_seed=42)
        eng.fit(prices)
        assert not eng.redundancy.isnull().all()
        assert (eng.redundancy >= 1.0).all()
        assert (eng.redundancy <= 3.0).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])