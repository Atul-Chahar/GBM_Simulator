"""
BTC GBM Prediction Model
========================
Geometric Brownian Motion simulator with FIGARCH volatility
and Student-t fat-tail modeling for Bitcoin hourly price prediction.
"""

from .data_fetcher import fetch_btc_klines, fetch_latest_bars
from .gbm_engine import GBMEngine
from .evaluator import evaluate_predictions, winkler_score
