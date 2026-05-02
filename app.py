"""
app.py — BTC Next-Hour Forecast Dashboard (Part B + C)
=======================================================
Live Streamlit dashboard for the AlphaI × Polaris challenge.

Features:
    - Real-time BTC price + 95% confidence interval prediction
    - Plotly chart showing last 50 bars with CI ribbon
    - Backtest metrics (coverage, avg width, Winkler) as headlines
    - Prediction history (Part C bonus)
    - Auto-refresh every 5 minutes
    - Bitcoin-themed dark mode

Usage:
    streamlit run app.py
"""

import json
import os
import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timezone, timedelta

from model.data_fetcher import fetch_latest_bars, get_close_prices
from model.gbm_engine import GBMEngine
from model.evaluator import evaluate_predictions, format_metrics_display
from persistence.storage import PredictionStore

# ── Page Configuration ───────────────────────────────────────────
st.set_page_config(
    page_title="BTC Next-Hour Forecast | AlphaI × Polaris",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    * { font-family: 'Inter', sans-serif; }

    /* Main header styling */
    .main-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #F7931A 0%, #FFD700 50%, #F7931A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #888;
        font-size: 0.9rem;
        font-weight: 300;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1A1F2E 0%, #0E1117 100%);
        border: 1px solid #2A2F3E;
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    }
    div[data-testid="stMetric"] label {
        color: #888 !important;
        font-size: 0.8rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F7931A !important;
    }

    /* Prediction box */
    .prediction-box {
        background: linear-gradient(135deg, #1a2332 0%, #0d1520 100%);
        border: 1px solid #F7931A40;
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(247, 147, 26, 0.1);
    }
    .prediction-box .price {
        font-size: 2.8rem;
        font-weight: 800;
        color: #FAFAFA;
        margin: 0.5rem 0;
    }
    .prediction-box .range {
        font-size: 1.4rem;
        font-weight: 600;
        color: #F7931A;
        margin: 0.5rem 0;
    }
    .prediction-box .label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 500;
    }
    .prediction-box .sublabel {
        font-size: 0.75rem;
        color: #666;
        margin-top: 0.3rem;
    }

    /* Regime badge */
    .regime-badge {
        display: inline-block;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #FAFAFA;
        border-bottom: 2px solid #F7931A40;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
    }

    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #555;
        font-size: 0.75rem;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Auto-refresh indicator */
    .refresh-indicator {
        text-align: right;
        color: #555;
        font-size: 0.7rem;
        padding: 0.2rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ── Cached Functions ─────────────────────────────────────────────

@st.cache_data(ttl=300)  # Cache for 5 minutes
def get_live_prediction():
    """
    Fetch latest data, run the model, return prediction.
    Cached for 5 minutes to avoid re-computation on every interaction.
    """
    # Fetch latest 500 closed bars (as brief specifies)
    df = fetch_latest_bars(num_bars=500)
    prices = get_close_prices(df)

    # Fit GBM model
    engine = GBMEngine(n_sims=10_000)
    engine.fit(prices)

    # Predict next-hour interval
    low, high, sims, mean_p = engine.predict_interval(confidence=0.95)

    # Get model info
    model_info = engine.get_model_info()

    # Compute chart ribbon for last 50 bars
    # Use the actual prediction width as reference, then scale each bar
    # proportionally by its relative sigma (from FIGARCH)
    pred_width = high - low
    latest_sigma = engine.sigma_fig.iloc[-1]

    chart_prices = prices.tail(50)
    sigma_vals = engine.sigma_fig.tail(50)
    chart_lows = []
    chart_highs = []
    for i, (idx, price) in enumerate(chart_prices.items()):
        if i < len(sigma_vals) and latest_sigma > 0:
            # Scale ribbon width proportionally to each bar's sigma
            sigma_ratio = sigma_vals.iloc[i] / latest_sigma
            half_w = (pred_width / 2) * sigma_ratio
        else:
            half_w = pred_width / 2
        chart_lows.append(price - half_w)
        chart_highs.append(price + half_w)

    return {
        "current_price": float(prices.iloc[-1]),
        "predicted_low": float(low),
        "predicted_high": float(high),
        "mean_prediction": float(mean_p),
        "model_info": model_info,
        "chart_dates": chart_prices.index.tolist(),
        "chart_prices": chart_prices.values.tolist(),
        "chart_lows": chart_lows,
        "chart_highs": chart_highs,
        "prediction_time": datetime.now(timezone.utc).isoformat(),
        "next_bar_close": _next_bar_close_time(),
    }


@st.cache_data(ttl=3600)  # Cache for 1 hour (backtest doesn't change)
def get_backtest_metrics():
    """
    Load and compute backtest metrics from backtest_results.jsonl.
    """
    filepath = "backtest_results.jsonl"
    if not os.path.exists(filepath):
        return None

    predictions = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                predictions.append(json.loads(line))

    if not predictions:
        return None

    metrics = evaluate_predictions(predictions)
    metrics["predictions_data"] = predictions
    return metrics


def _next_bar_close_time() -> str:
    """Calculate when the next hourly bar closes."""
    now = datetime.now(timezone.utc)
    next_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    remaining = next_hour - now
    minutes = int(remaining.total_seconds() // 60)
    return f"{minutes} min"


# ── Main App ─────────────────────────────────────────────────────

def main():
    # ── Header ───────────────────────────────────────────────────
    st.markdown("""
    <div class="main-header">
        <h1>₿ BTC Next-Hour Forecast</h1>
        <p>AlphaI × Polaris Challenge — GBM + FIGARCH + Student-t Model</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load backtest metrics ────────────────────────────────────
    backtest = get_backtest_metrics()

    if backtest:
        formatted = format_metrics_display(backtest)

        # Headline metrics row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Coverage (95%)", formatted["coverage"])
        with col2:
            st.metric("Avg Width", formatted["avg_width"])
        with col3:
            st.metric("Winkler Score", formatted["winkler"])
        with col4:
            st.metric("Backtest Bars", formatted["predictions"])
    else:
        st.warning(
            "⚠️ Backtest results not found. "
            "Run `python backtest.py` to generate `backtest_results.jsonl`."
        )

    # ── Live Prediction ──────────────────────────────────────────
    try:
        with st.spinner("🔮 Running GBM model on latest 500 bars..."):
            prediction = get_live_prediction()

        current = prediction["current_price"]
        low = prediction["predicted_low"]
        high = prediction["predicted_high"]
        width = high - low
        regime = prediction["model_info"].get("volatility_regime", "Unknown")

        # Prediction display
        st.markdown(f"""
        <div class="prediction-box">
            <div class="label">Current BTC Price</div>
            <div class="price">${current:,.2f}</div>
            <div class="label">Predicted 95% Range for Next Hour</div>
            <div class="range">${low:,.2f} — ${high:,.2f}</div>
            <div class="sublabel">Width: ${width:,.2f} | Next bar closes in: {prediction['next_bar_close']}</div>
            <div class="sublabel">Volatility Regime: {regime}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Chart ────────────────────────────────────────────────
        st.markdown('<div class="section-header">📊 Price Chart — Last 50 Bars with 95% Confidence Ribbon</div>', unsafe_allow_html=True)

        fig = go.Figure()

        dates = prediction["chart_dates"]
        prices_chart = prediction["chart_prices"]
        lows_chart = prediction["chart_lows"]
        highs_chart = prediction["chart_highs"]

        # Upper band (invisible, for fill)
        fig.add_trace(go.Scatter(
            x=dates,
            y=highs_chart,
            mode='lines',
            line=dict(width=0),
            showlegend=False,
            hoverinfo='skip',
        ))

        # Lower band (with fill to upper)
        fig.add_trace(go.Scatter(
            x=dates,
            y=lows_chart,
            mode='lines',
            line=dict(width=0),
            fill='tonexty',
            fillcolor='rgba(247, 147, 26, 0.15)',
            name='95% Confidence',
            hoverinfo='skip',
        ))

        # Price line
        fig.add_trace(go.Scatter(
            x=dates,
            y=prices_chart,
            mode='lines+markers',
            line=dict(color='#F7931A', width=2),
            marker=dict(size=3, color='#F7931A'),
            name='BTC Price',
            hovertemplate='%{x}<br>Price: $%{y:,.2f}<extra></extra>',
        ))

        # Next-bar prediction range (shaded area extending right)
        if dates:
            last_date = dates[-1]
            next_date = last_date + pd.Timedelta(hours=1)

            fig.add_trace(go.Scatter(
                x=[last_date, next_date, next_date, last_date],
                y=[current, high, low, current],
                fill='toself',
                fillcolor='rgba(0, 255, 136, 0.2)',
                line=dict(color='rgba(0, 255, 136, 0.5)', width=1),
                name=f'Next Hour: ${low:,.0f}–${high:,.0f}',
                hoverinfo='skip',
            ))

            # Dashed lines for predicted range
            fig.add_hline(y=high, line_dash="dash", line_color="#00FF88",
                         annotation_text=f"Upper: ${high:,.2f}",
                         annotation_position="top right",
                         annotation_font_color="#00FF88")
            fig.add_hline(y=low, line_dash="dash", line_color="#00FF88",
                         annotation_text=f"Lower: ${low:,.2f}",
                         annotation_position="bottom right",
                         annotation_font_color="#00FF88")

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(14,17,23,1)',
            height=450,
            margin=dict(l=50, r=50, t=30, b=50),
            xaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                title="Time (UTC)",
            ),
            yaxis=dict(
                gridcolor='rgba(255,255,255,0.05)',
                title="Price (USDT)",
                tickformat="$,.0f",
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=11),
            ),
            hovermode="x unified",
        )

        st.plotly_chart(fig, width='stretch')

        # ── Model Details (collapsible) ──────────────────────────
        with st.expander("🔧 Model Parameters"):
            info = prediction["model_info"]
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Hourly Drift (μ):** {info.get('hourly_drift', 0):.8f}")
                st.markdown(f"**Student-t ν:** {info.get('nu_degrees_freedom', 0):.2f}")
            with col2:
                st.markdown(f"**Latest σ:** {info.get('latest_sigma', 0):.6f}")
                st.markdown(f"**Mean σ²:** {info.get('mean_sigma2', 0):.10f}")
            with col3:
                st.markdown(f"**Regime:** {info.get('volatility_regime', 'N/A')}")
                st.markdown(f"**Simulations:** 10,000")

        # ── Part C: Prediction History ───────────────────────────
        st.markdown('<div class="section-header">📜 Prediction History</div>', unsafe_allow_html=True)

        store = PredictionStore(use_gsheets=False)

        # Save this prediction if enough time has passed
        if store.should_save_new_prediction():
            store.save_prediction({
                "timestamp": prediction["prediction_time"],
                "current_price": current,
                "predicted_low_95": low,
                "predicted_high_95": high,
            })

        # Display history
        history_df = store.get_history_dataframe()
        if len(history_df) > 0:
            st.dataframe(
                history_df.style.format({
                    "current_price": "${:,.2f}",
                    "predicted_low_95": "${:,.2f}",
                    "predicted_high_95": "${:,.2f}",
                    "actual_close": lambda x: f"${x:,.2f}" if pd.notna(x) and x else "⏳",
                    "winkler": lambda x: f"{x:,.2f}" if pd.notna(x) and x else "—",
                }),
                width='stretch',
                hide_index=True,
            )
            st.caption(f"Total predictions stored: {len(history_df)}")
        else:
            st.info("No prediction history yet. Each visit generates a new prediction that gets saved here.")

    except Exception as e:
        st.error(f"❌ Error running model: {str(e)}")
        st.exception(e)

    # ── Footer ───────────────────────────────────────────────────
    st.markdown("""
    <div class="footer">
        <p>
            Built for the AlphaI × Polaris Challenge |
            Model: GBM + FIGARCH(1,0,1) + Student-t |
            Data: Binance BTCUSDT 1H |
            Auto-refreshes every 5 minutes
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Auto-refresh ─────────────────────────────────────────────
    # Refresh the page every 5 minutes for live updates
    st.markdown(
        f"""
        <div class="refresh-indicator">
            Last updated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
