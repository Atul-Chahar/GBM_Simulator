"""
gbm_engine.py — Geometric Brownian Motion Simulation Engine
============================================================
Adapted from the AlphaI × Polaris starter notebook.

Core model: GBM with FIGARCH(1,0,1) conditional volatility,
entropy-based crisis detection, and Student-t fat-tail innovations.

Key concepts (from the brief):
    1. NO PEEKING — At bar N, only use data up to bar N-1
    2. VOLATILITY CLUSTERING — Recent volatility drives range width
    3. FAT TAILS — Student-t distribution, never Normal

Changes from starter:
    - Adapted from daily forex (USDCHF) to hourly crypto (BTCUSDT)
    - dt = 1 (one period = one hour; FIGARCH is fitted on hourly returns,
      so volatility is already in per-hour units)
    - Removed global variable dependencies (redundancy, info_filter)
    - Added proper parameter encapsulation
    - Removed option pricing code (not needed for this challenge)
    - Fixed pip install statement (was bare Python, not notebook magic)
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from arch import arch_model
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


class GBMEngine:
    """
    Geometric Brownian Motion engine with FIGARCH volatility modeling.

    Usage:
        engine = GBMEngine()
        engine.fit(close_prices)  # pd.Series of close prices
        low, high = engine.predict_interval(confidence=0.95)
    """

    DT = 1.0
    N_SIMS = 10_000
    ENTROPY_WINDOW = 60
    MOMENTUM_WINDOW = 60

    BASE_PARAMS = {
        "alpha": 0.15,
        "delta": 0.10,
        "gamma": 0.2,
        "kappa": 0.1,
        "eta": 1e-3,
    }

    def __init__(self, n_sims: int = None, entropy_window: int = None, random_seed: int = 42,
                 calibrate: bool = True):
        self.n_sims = n_sims or self.N_SIMS
        self.entropy_window = entropy_window or self.ENTROPY_WINDOW
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        self._calibrate = calibrate

        self._fitted = False
        self.mu = None
        self.S0 = None
        self.nu = None
        self.sigma_fig = None
        self.H_series = None
        self.M_series = None
        self.bar_sigma2 = None
        self.redundancy = None
        self.info_filter = None
        self.params = None
        self._model_type = "GARCH"
        self._cal_factor = 0.85
        self._selected_order = None

    def fit(self, close_prices: pd.Series) -> "GBMEngine":
        prices = close_prices.copy()

        log_ret = np.log(prices / prices.shift(1)).dropna()
        self.mu = log_ret.mean()
        self.S0 = prices.iloc[-1]

        self._fit_volatility_model(log_ret)

        self.H_series = self._rolling_entropy(
            self._get_residuals(log_ret), window=self.entropy_window
        )
        self.M_series = log_ret.abs().rolling(self.MOMENTUM_WINDOW).mean()

        sigma_sq = self.sigma_fig ** 2
        self.bar_sigma2 = sigma_sq.mean()

        price_var_5 = prices.rolling(5).var()
        price_var_20 = prices.rolling(20).var()
        ratio = price_var_5 / price_var_20.replace(0, np.nan)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(0)
        self.redundancy = 1 + 0.02 * np.log1p(ratio)

        H_mean = self.H_series.mean()
        self.info_filter = (self.H_series > H_mean).astype(float)

        self.params = self.BASE_PARAMS.copy()
        H_max = max(self.H_series.max(), 1e-10)
        M_max = max(self.M_series.max(), 1e-10)
        alpha0 = self.params["alpha"]
        delta0 = self.params["delta"]

        if alpha0 * H_max + delta0 * M_max >= 1:
            fac = 0.95 / (alpha0 * H_max + delta0 * M_max)
            self.params["alpha"] *= fac
            self.params["delta"] *= fac

        self._calibrate_factor(prices, log_ret)

        self._fitted = True
        return self

    def _fit_volatility_model(self, log_ret: pd.Series) -> None:
        orders = [
            (1, 0, 1),
            (1, 0, 2),
            (1, 1, 1),
        ]

        best_aic = None
        best_res = None
        best_model_type = "GARCH"
        best_order = None

        for p, o, q in orders:
            try:
                am = arch_model(
                    log_ret * 100,
                    vol="FIGARCH",
                    p=p, o=o, q=q,
                    dist="studentst",
                )
                res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
                if best_aic is None or res.aic < best_aic:
                    best_aic = res.aic
                    best_res = res
                    best_model_type = "FIGARCH"
                    best_order = f"FIGARCH({p},{o},{q})"
            except Exception:
                pass

        if best_res is None:
            for p, o, q in [(1, 0, 1), (1, 0, 2), (2, 0, 2)]:
                try:
                    am = arch_model(
                        log_ret * 100,
                        vol="GARCH",
                        p=p, o=o, q=q,
                        dist="studentst",
                    )
                    res = am.fit(disp="off", show_warning=False, options={"maxiter": 200})
                    if best_aic is None or res.aic < best_aic:
                        best_aic = res.aic
                        best_res = res
                        best_model_type = "GARCH"
                        best_order = f"GARCH({p},{o},{q})"
                except Exception:
                    pass

        self._model_type = best_model_type
        self._selected_order = best_order
        self.sigma_fig = best_res.conditional_volatility / 100

        cond_vol = best_res.conditional_volatility
        cond_vol = cond_vol.replace(0, np.nan).dropna()
        resid = (log_ret.loc[cond_vol.index] * 100 - best_res.params["mu"]) / cond_vol

        try:
            self.nu = max(4, stats.t.fit(resid.dropna(), floc=0, fscale=1)[0])
        except Exception:
            self.nu = 5.0

    def _get_residuals(self, log_ret: pd.Series) -> pd.Series:
        cond_vol = self.sigma_fig * 100
        cond_vol = cond_vol.replace(0, np.nan).dropna()
        return log_ret.loc[cond_vol.index] * 100 / cond_vol

    def _calibrate_factor(self, prices: pd.Series, log_ret: pd.Series) -> None:
        if not self._calibrate:
            self._cal_factor = 0.85
            return

        cal_window = min(24, len(prices) // 4)
        if cal_window < 10:
            self._cal_factor = 0.85
            return

        target_coverage = 0.95
        best_factor = 0.85

        try:
            train_end_idx = len(prices) - cal_window
            if train_end_idx < 60:
                self._cal_factor = 0.85
                return

            train_prices = prices.iloc[:train_end_idx]
            test_prices = prices.iloc[train_end_idx:]

            temp_engine = GBMEngine(n_sims=1000, random_seed=42, calibrate=False)
            temp_engine.fit(train_prices)

            hits = 0
            total = 0
            for i in range(len(test_prices) - 1):
                idx = train_end_idx + i
                lookback = prices.iloc[max(0, idx - 500):idx]
                if len(lookback) < 60:
                    continue

                te = GBMEngine(n_sims=1000, random_seed=42, calibrate=False)
                te.fit(lookback)
                low, high, _, _ = te.predict_interval(confidence=0.95)
                actual = float(prices.iloc[idx])
                if low <= actual <= high:
                    hits += 1
                total += 1

            if total > 0:
                observed = hits / total
                if observed > target_coverage + 0.03:
                    best_factor = 0.75
                elif observed > target_coverage + 0.01:
                    best_factor = 0.80
                elif observed < target_coverage - 0.03:
                    best_factor = 0.92
                elif observed < target_coverage - 0.01:
                    best_factor = 0.88

        except Exception:
            best_factor = 0.85

        self._cal_factor = max(0.5, min(1.5, best_factor))

    def predict_interval(
        self, confidence: float = 0.95, n_steps: int = 1
    ) -> tuple:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_interval()")

        alpha = 1 - confidence
        paths = self._simulate_mc(n_steps=n_steps)

        terminal_prices = paths[:, -1]

        low = np.percentile(terminal_prices, (alpha / 2) * 100)
        high = np.percentile(terminal_prices, (1 - alpha / 2) * 100)
        mean_price = terminal_prices.mean()

        return low, high, terminal_prices, mean_price

    def predict_range_for_bars(
        self, n_bars: int = 50, confidence: float = 0.95
    ) -> list:
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_range_for_bars()")

        alpha = 1 - confidence
        z_low = stats.t.ppf(alpha / 2, df=self.nu)
        z_high = stats.t.ppf(1 - alpha / 2, df=self.nu)

        scale = np.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1.0

        ranges = []
        sigma_vals = self.sigma_fig.tail(n_bars)

        for sigma in sigma_vals:
            half_width = sigma * scale * abs(z_high) * np.sqrt(self.DT)
            ranges.append(half_width)

        return ranges

    def _simulate_mc(self, n_steps: int = 1) -> np.ndarray:
        out = np.zeros((self.n_sims, n_steps + 1))

        for i in range(self.n_sims):
            path, _ = self._simulate_single_path(n_steps)
            out[i] = path

        return out

    def _simulate_single_path(self, n_steps: int) -> tuple:
        S = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)
        S[0] = self.S0

        sigma2 = self.sigma_fig.iloc[-1] ** 2
        params = self.params.copy()

        H_max = max(self.H_series.max(), 1e-10)
        M_max = max(self.M_series.max(), 1e-10)

        H_val_raw = self.H_series.iloc[-1] if not np.isnan(self.H_series.iloc[-1]) else 0
        M_val_raw = self.M_series.iloc[-1] if not np.isnan(self.M_series.iloc[-1]) else 0
        redundancy_val = self.redundancy.iloc[-1] if not np.isnan(self.redundancy.iloc[-1]) else 1.0

        for t in range(1, n_steps + 1):
            H_val = min(H_val_raw / H_max, 1.0)
            M_val = min(M_val_raw / M_max, 1.0)

            crisis = (H_val > 0.8) or (M_val > 0.8)
            delta_t = params["delta"] if crisis else 0.0

            sigma2 = (
                sigma2
                * (1 + params["alpha"] * H_val + delta_t * M_val)
                + params["gamma"] * (self.bar_sigma2 - sigma2)
            )

            sigma2 *= max(1e-12, redundancy_val)

            sigma2 *= self._cal_factor

            sigma2 = max(1e-12, min(sigma2, 0.5))

            Z = self.rng.standard_t(self.nu) * np.sqrt(
                (self.nu - 2) / self.nu
            )

            S[t] = S[t - 1] * np.exp(
                (self.mu - 0.5 * sigma2) * self.DT
                + np.sqrt(sigma2 * self.DT) * Z
            )
            V[t] = sigma2

            params = self._update_params(params, sigma2, t)

        return S, V

    def _update_params(self, params: dict, sigma2: float, t: int) -> dict:
        err = sigma2 - self.bar_sigma2
        lr = params["eta"] / (1 + t ** 0.55)
        params["gamma"] = np.clip(params["gamma"] + lr * err, 0.01, 0.5)
        return params

    @staticmethod
    def _rolling_entropy(
        x: pd.Series, window: int = 60, bins: int = 20
    ) -> pd.Series:
        def _entropy(v):
            p, _ = np.histogram(v, bins=bins, density=True)
            p = p[p > 0]
            return -np.sum(p * np.log(p))

        return x.rolling(window).apply(_entropy, raw=True)

    def get_volatility_regime(self) -> str:
        if not self._fitted:
            return "Unknown"

        H_max = max(self.H_series.max(), 1e-10)
        M_max = max(self.M_series.max(), 1e-10)

        H_val = self.H_series.iloc[-1] / H_max if not np.isnan(self.H_series.iloc[-1]) else 0
        M_val = self.M_series.iloc[-1] / M_max if not np.isnan(self.M_series.iloc[-1]) else 0

        if H_val > 0.8 or M_val > 0.8:
            return "🔴 High Volatility"
        elif H_val > 0.5 or M_val > 0.5:
            return "🟡 Moderate"
        else:
            return "🟢 Calm"

    def get_model_info(self) -> dict:
        if not self._fitted:
            return {}

        tag = self._model_type
        if self._model_type == "GARCH":
            model_label = f"{tag} + STUDENT-T"
        else:
            model_label = f"{self._selected_order} + STUDENT-T"

        return {
            "current_price": float(self.S0),
            "hourly_drift": float(self.mu),
            "nu_degrees_freedom": float(self.nu),
            "latest_sigma": float(self.sigma_fig.iloc[-1]),
            "mean_sigma2": float(self.bar_sigma2),
            "volatility_regime": self.get_volatility_regime(),
            "model_type": tag,
            "model_label": model_label,
        }


if __name__ == "__main__":
    from data_fetcher import fetch_latest_bars, get_close_prices

    print("Fetching 500 bars...")
    df = fetch_latest_bars(500)
    prices = get_close_prices(df)
    print(f"Got {len(prices)} bars. Latest: ${prices.iloc[-1]:,.2f}")

    print("\nFitting GBM model...")
    engine = GBMEngine(n_sims=1000, random_seed=42)
    engine.fit(prices)

    print(f"Model type: {engine._model_type}")
    print(f"Calibration factor: {engine._cal_factor:.4f}")

    print("\nPredicting next-hour interval...")
    low, high, sims, mean_p = engine.predict_interval()
    print(f"Current: ${engine.S0:,.2f}")
    print(f"95% CI:  ${low:,.2f} — ${high:,.2f}")
    print(f"Width:   ${high - low:,.2f}")
    print(f"Regime:  {engine.get_volatility_regime()}")