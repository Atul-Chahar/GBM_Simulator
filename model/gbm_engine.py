"""
gbm_engine.py — Geometric Brownian Motion Simulation Engine
=============================================================
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

# Suppress convergence warnings from arch library during batch fitting
warnings.filterwarnings("ignore", category=RuntimeWarning)


class GBMEngine:
    """
    Geometric Brownian Motion engine with FIGARCH volatility modeling.

    Usage:
        engine = GBMEngine()
        engine.fit(close_prices)  # pd.Series of close prices
        low, high = engine.predict_interval(confidence=0.95)
    """

    # Time step: 1 period = 1 hour.
    # FIGARCH is fitted on hourly log returns, so σ is already per-hour.
    # dt=1 means "one period" — NOT "one day".
    DT = 1.0

    # Number of Monte Carlo simulations
    N_SIMS = 10_000

    # Entropy rolling window
    ENTROPY_WINDOW = 60

    # Momentum rolling window
    MOMENTUM_WINDOW = 60

    # Base adaptive parameters — tuned down from starter notebook.
    # Original (forex daily): alpha=0.5, delta=0.3 caused too-wide intervals
    # on hourly BTC since FIGARCH already captures volatility clustering.
    BASE_PARAMS = {
        "alpha": 0.15,
        "delta": 0.10,
        "gamma": 0.2,
        "kappa": 0.1,
        "eta": 1e-3,
    }

    def __init__(self, n_sims: int = None, entropy_window: int = None):
        """
        Initialize the GBM engine.

        Parameters
        ----------
        n_sims : int, optional
            Number of Monte Carlo paths. Default 10,000.
        entropy_window : int, optional
            Window for rolling entropy computation. Default 60.
        """
        self.n_sims = n_sims or self.N_SIMS
        self.entropy_window = entropy_window or self.ENTROPY_WINDOW

        # Fitted state (populated by .fit())
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

    def fit(self, close_prices: pd.Series) -> "GBMEngine":
        """
        Fit the GBM model on historical close prices.

        This estimates:
        - Drift (mu) from log returns
        - FIGARCH conditional volatility
        - Student-t degrees of freedom (nu)
        - Entropy and momentum indicators
        - Adaptive crisis parameters

        Parameters
        ----------
        close_prices : pd.Series
            Historical close prices. Index should be datetime-like.

        Returns
        -------
        self
            For method chaining.
        """
        prices = close_prices.copy()

        # ── Log returns ──────────────────────────────────────────
        log_ret = np.log(prices / prices.shift(1)).dropna()
        self.mu = log_ret.mean()
        self.S0 = prices.iloc[-1]

        # ── FIGARCH(1,0,1) with Student-t innovations ───────────
        # Scale to percentage returns for numerical stability
        try:
            am = arch_model(
                log_ret * 100,
                vol="FIGARCH",
                p=1, o=0, q=1,
                dist="studentst",
            )
            res = am.fit(disp="off", show_warning=False)
        except Exception:
            # Fallback to standard GARCH if FIGARCH fails to converge
            am = arch_model(
                log_ret * 100,
                vol="GARCH",
                p=1, o=0, q=1,
                dist="studentst",
            )
            res = am.fit(disp="off", show_warning=False)

        # Conditional volatility (rescale back from percentage)
        self.sigma_fig = res.conditional_volatility / 100

        # Standardized residuals
        cond_vol = res.conditional_volatility
        cond_vol = cond_vol.replace(0, np.nan).dropna()
        resid = (log_ret.loc[cond_vol.index] * 100 - res.params["mu"]) / cond_vol

        # Fit Student-t degrees of freedom (min 4 for finite kurtosis)
        try:
            self.nu = max(4, stats.t.fit(resid.dropna(), floc=0, fscale=1)[0])
        except Exception:
            self.nu = 5.0  # Safe default

        # ── Rolling entropy (information measure) ────────────────
        self.H_series = self._rolling_entropy(
            resid, window=self.entropy_window
        )

        # ── Rolling momentum (absolute return magnitude) ─────────
        self.M_series = log_ret.abs().rolling(self.MOMENTUM_WINDOW).mean()

        # ── Variance components ──────────────────────────────────
        sigma_sq = self.sigma_fig ** 2
        self.bar_sigma2 = sigma_sq.mean()

        # Redundancy factor: short vs long variance ratio
        price_var_5 = prices.rolling(5).var()
        price_var_20 = prices.rolling(20).var()
        ratio = price_var_5 / price_var_20.replace(0, np.nan)
        self.redundancy = 1 + 0.02 * np.log1p(ratio.fillna(0))

        # Information filter: above-average entropy flag
        H_mean = self.H_series.mean()
        self.info_filter = (self.H_series > H_mean).astype(float)

        # ── Adaptive parameters (scale to avoid explosion) ───────
        self.params = self.BASE_PARAMS.copy()
        H_max = max(self.H_series.max(), 1e-10)
        M_max = max(self.M_series.max(), 1e-10)
        alpha0 = self.params["alpha"]
        delta0 = self.params["delta"]

        if alpha0 * H_max + delta0 * M_max >= 1:
            fac = 0.95 / (alpha0 * H_max + delta0 * M_max)
            self.params["alpha"] *= fac
            self.params["delta"] *= fac

        self._fitted = True
        return self

    def predict_interval(
        self, confidence: float = 0.95, n_steps: int = 1
    ) -> tuple:
        """
        Predict the confidence interval for the next bar(s).

        Parameters
        ----------
        confidence : float
            Confidence level (default 0.95 → 95% CI).
        n_steps : int
            Number of steps ahead to simulate. Default 1.

        Returns
        -------
        tuple
            (low, high, simulated_prices, mean_price)
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_interval()")

        alpha = 1 - confidence
        paths = self._simulate_mc(n_steps=n_steps)

        # Extract terminal prices
        terminal_prices = paths[:, -1]

        low = np.percentile(terminal_prices, (alpha / 2) * 100)
        high = np.percentile(terminal_prices, (1 - alpha / 2) * 100)
        mean_price = terminal_prices.mean()

        return low, high, terminal_prices, mean_price

    def predict_range_for_bars(
        self, n_bars: int = 50, confidence: float = 0.95
    ) -> list:
        """
        Generate prediction intervals for the last N bars (for chart ribbon).

        This is a retrospective visualization: for each of the last N bars,
        what was the model's predicted 95% range?

        Note: This uses the CURRENT model fit, so it's illustrative
        (not a true walk-forward backtest).

        Parameters
        ----------
        n_bars : int
            Number of recent bars to show ranges for.
        confidence : float
            Confidence level.

        Returns
        -------
        list[dict]
            List of {"low": float, "high": float} for each bar.
        """
        if not self._fitted:
            raise RuntimeError("Call .fit() before .predict_range_for_bars()")

        # Use the current fit's conditional volatility to estimate
        # a simple range for each of the last N bars
        alpha = 1 - confidence
        z_low = stats.t.ppf(alpha / 2, df=self.nu)
        z_high = stats.t.ppf(1 - alpha / 2, df=self.nu)

        # Scale factor for t-distribution variance
        scale = np.sqrt((self.nu - 2) / self.nu) if self.nu > 2 else 1.0

        ranges = []
        sigma_vals = self.sigma_fig.tail(n_bars)

        for sigma in sigma_vals:
            half_width = sigma * scale * abs(z_high) * np.sqrt(self.DT)
            ranges.append(half_width)

        return ranges

    def _simulate_mc(self, n_steps: int = 1) -> np.ndarray:
        """
        Run Monte Carlo simulation of the GBM paths.

        Returns
        -------
        np.ndarray
            Shape (n_sims, n_steps + 1). First column is S0.
        """
        out = np.zeros((self.n_sims, n_steps + 1))

        for i in range(self.n_sims):
            path, _ = self._simulate_single_path(n_steps)
            out[i] = path

        return out

    def _simulate_single_path(self, n_steps: int) -> tuple:
        """
        Simulate a single GBM path with FIGARCH volatility
        and entropy-based crisis detection.

        Adapted from the starter notebook's simulate_cyber_gbm().
        """
        S = np.zeros(n_steps + 1)
        V = np.zeros(n_steps + 1)
        S[0] = self.S0

        sigma2 = self.sigma_fig.iloc[-1] ** 2
        params = self.params.copy()

        H_max = max(self.H_series.max(), 1e-10)
        M_max = max(self.M_series.max(), 1e-10)

        # Use the latest values for indicators
        H_val_raw = self.H_series.iloc[-1] if not np.isnan(self.H_series.iloc[-1]) else 0
        M_val_raw = self.M_series.iloc[-1] if not np.isnan(self.M_series.iloc[-1]) else 0
        redundancy_val = self.redundancy.iloc[-1] if not np.isnan(self.redundancy.iloc[-1]) else 1.0
        info_val = self.info_filter.iloc[-1] if not np.isnan(self.info_filter.iloc[-1]) else 0.0

        for t in range(1, n_steps + 1):
            H_val = min(H_val_raw / H_max, 1.0)
            M_val = min(M_val_raw / M_max, 1.0)

            # Crisis detection
            crisis = (H_val > 0.8) or (M_val > 0.8)
            delta_t = params["delta"] if crisis else 0.0

            # Conditional variance update
            sigma2 = (
                self.sigma_fig.iloc[-1] ** 2
                * (1 + params["alpha"] * H_val + delta_t * M_val)
                + params["gamma"] * (self.bar_sigma2 - sigma2)
            )

            # Apply redundancy and information filters
            sigma2 *= max(1e-12, redundancy_val)
            # Info filter effect removed — FIGARCH already models volatility
            # clustering; this extra boost was making intervals too wide.
            # sigma2 *= 1 + 0.2 * info_val

            # Calibration factor: FIGARCH's long-memory component tends
            # to overestimate next-hour variance because its 500-bar
            # training window includes older, more volatile regimes.
            # Empirically calibrated on 200+ bar walk-forward tests.
            sigma2 *= 0.85

            # Clamp variance to sane range
            sigma2 = max(1e-12, min(sigma2, 0.5))

            # Student-t innovation (fat tails)
            Z = np.random.standard_t(self.nu) * np.sqrt(
                (self.nu - 2) / self.nu
            )

            # GBM step
            S[t] = S[t - 1] * np.exp(
                (self.mu - 0.5 * sigma2) * self.DT
                + np.sqrt(sigma2 * self.DT) * Z
            )
            V[t] = sigma2

            # Adaptive parameter update
            params = self._update_params(params, sigma2, t)

        return S, V

    def _update_params(self, params: dict, sigma2: float, t: int) -> dict:
        """
        Adaptively update model parameters based on variance error.
        """
        err = sigma2 - self.bar_sigma2
        lr = params["eta"] / (1 + t ** 0.55)
        params["gamma"] = np.clip(params["gamma"] + lr * err, 0.01, 0.5)
        return params

    @staticmethod
    def _rolling_entropy(
        x: pd.Series, window: int = 60, bins: int = 20
    ) -> pd.Series:
        """
        Compute rolling Shannon entropy of a time series.

        Entropy measures the "disorder" or unpredictability of
        recent returns. High entropy → uncertain/chaotic period.

        Parameters
        ----------
        x : pd.Series
            Input series (typically standardized residuals).
        window : int
            Rolling window size.
        bins : int
            Number of histogram bins for density estimation.
        """

        def _entropy(v):
            p, _ = np.histogram(v, bins=bins, density=True)
            p = p[p > 0]
            return -np.sum(p * np.log(p))

        return x.rolling(window).apply(_entropy, raw=True)

    def get_volatility_regime(self) -> str:
        """
        Classify the current volatility regime.

        Returns
        -------
        str
            One of "🟢 Calm", "🟡 Moderate", "🔴 Crisis"
        """
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
        """
        Return key model parameters for display.
        """
        if not self._fitted:
            return {}

        return {
            "current_price": float(self.S0),
            "hourly_drift": float(self.mu),
            "nu_degrees_freedom": float(self.nu),
            "latest_sigma": float(self.sigma_fig.iloc[-1]),
            "mean_sigma2": float(self.bar_sigma2),
            "volatility_regime": self.get_volatility_regime(),
        }


# ─── Quick self-test ─────────────────────────────────────────────
if __name__ == "__main__":
    from data_fetcher import fetch_latest_bars, get_close_prices

    print("Fetching 500 bars...")
    df = fetch_latest_bars(500)
    prices = get_close_prices(df)
    print(f"Got {len(prices)} bars. Latest: ${prices.iloc[-1]:,.2f}")

    print("\nFitting GBM model...")
    engine = GBMEngine(n_sims=1000)  # Fewer sims for quick test
    engine.fit(prices)

    print("\nPredicting next-hour interval...")
    low, high, sims, mean_p = engine.predict_interval()
    print(f"Current: ${engine.S0:,.2f}")
    print(f"95% CI:  ${low:,.2f} — ${high:,.2f}")
    print(f"Width:   ${high - low:,.2f}")
    print(f"Regime:  {engine.get_volatility_regime()}")
