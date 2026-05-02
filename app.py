"""
app.py — BTC Next-Hour Forecast Dashboard
==========================================
xAI-inspired brutalist minimalism + trading dashboard density.
JetBrains Mono display type. Dark/light theme toggle.
"""

import json, os, numpy as np, pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from datetime import datetime, timezone, timedelta
from model.data_fetcher import fetch_latest_bars, get_close_prices
from model.gbm_engine import GBMEngine
from model.evaluator import evaluate_predictions, format_metrics_display
from persistence.storage import PredictionStore

# ── Page Config ──────────────────────────────────────────
st.set_page_config(
    page_title="BTC Forecast — AlphaI × Polaris",
    page_icon="₿", layout="wide", initial_sidebar_state="collapsed",
)

# ── Theme State ──────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def toggle_theme():
    st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

is_dark = st.session_state.theme == "dark"

# ── Load + Inject CSS ────────────────────────────────────
css_path = os.path.join(os.path.dirname(__file__), "static", "styles.css")
css = open(css_path).read() if os.path.exists(css_path) else ""
st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Theme-specific overrides
if is_dark:
    st.markdown("""<style>
    .stApp{background:#1f2228!important;color:#fff!important}
    .metric-card,.bottom-panel{background:#1f2228!important}
    .metrics-grid,.bottom-grid,.stats-grid{background:rgba(255,255,255,0.1)!important}
    </style>""", unsafe_allow_html=True)
else:
    st.markdown("""<style>
    .stApp{background:#fafafa!important;color:#1f2228!important}
    .metric-card,.bottom-panel{background:#fafafa!important}
    .metrics-grid,.bottom-grid,.stats-grid{background:rgba(0,0,0,0.1)!important}
    .hero-price,.metric-value,.pred-range-value{color:#1f2228!important}
    .nav-bar,.site-footer{border-color:rgba(0,0,0,0.1)!important}
    .prediction-card,.chart-section{border-color:rgba(0,0,0,0.1)!important;
        background:rgba(0,0,0,0.02)!important}
    .nav-tag{border-color:rgba(0,0,0,0.15)!important;color:rgba(0,0,0,0.5)!important}
    .hero-label,.metric-label,.pred-range-label,.panel-title,.chart-tag,
    .pred-detail-label{color:rgba(0,0,0,0.4)!important}
    .hero-subtitle,.metric-sub,.param-key,.footer-text,.footer-mono,
    .chart-title{color:rgba(0,0,0,0.5)!important}
    .param-val,.pred-detail-value{color:#1f2228!important}
    .nav-brand{color:#1f2228!important}
    .param-row{border-color:rgba(0,0,0,0.08)!important}
    .param-row:hover{background:rgba(0,0,0,0.03)!important}
    </style>""", unsafe_allow_html=True)


# ── Cached Data Functions ────────────────────────────────
@st.cache_data(ttl=300)
def get_live_data():
    df = fetch_latest_bars(num_bars=500)
    prices = get_close_prices(df)
    engine = GBMEngine(n_sims=10_000)
    engine.fit(prices)
    low, high, sims, mean_p = engine.predict_interval(confidence=0.95)
    info = engine.get_model_info()

    # Chart ribbon
    pw = high - low
    ls = engine.sigma_fig.iloc[-1]
    cp = prices.tail(50)
    sv = engine.sigma_fig.tail(50)
    cl, ch = [], []
    for i, (_, p) in enumerate(cp.items()):
        r = sv.iloc[i] / ls if i < len(sv) and ls > 0 else 1
        hw = (pw / 2) * r
        cl.append(p - hw); ch.append(p + hw)

    # Extra analytics
    log_ret = np.log(prices / prices.shift(1)).dropna()
    vol_series = engine.sigma_fig

    # Hourly returns for bar chart (last 24)
    recent_ret = log_ret.tail(24)

    # Price stats
    p24h = prices.tail(24)
    p7d = prices.tail(168) if len(prices) >= 168 else prices
    high_24h = p24h.max()
    low_24h = p24h.min()
    change_24h = (p24h.iloc[-1] - p24h.iloc[0]) / p24h.iloc[0] * 100
    high_7d = p7d.max()
    low_7d = p7d.min()
    change_7d = (p7d.iloc[-1] - p7d.iloc[0]) / p7d.iloc[0] * 100
    avg_vol_24h = log_ret.tail(24).std()
    avg_vol_7d = log_ret.tail(168).std() if len(log_ret) >= 168 else log_ret.std()

    return {
        "current_price": float(prices.iloc[-1]),
        "predicted_low": float(low), "predicted_high": float(high),
        "mean_prediction": float(mean_p), "model_info": info,
        "chart_dates": cp.index.tolist(), "chart_prices": cp.values.tolist(),
        "chart_lows": cl, "chart_highs": ch,
        "prediction_time": (prices.index[-1] + pd.Timedelta(hours=1)).isoformat(),
        # Extra data
        "log_returns": log_ret.tail(100).values.tolist(),
        "log_ret_dates": log_ret.tail(100).index.tolist(),
        "vol_series": vol_series.tail(50).values.tolist(),
        "vol_dates": vol_series.tail(50).index.tolist(),
        "recent_ret": recent_ret.values.tolist(),
        "recent_ret_dates": [d.strftime("%H:%M") for d in recent_ret.index],
        "sims_terminal": sims.tolist()[:2000],  # subsample for histogram
        "high_24h": float(high_24h), "low_24h": float(low_24h),
        "change_24h": float(change_24h),
        "high_7d": float(high_7d), "low_7d": float(low_7d),
        "change_7d": float(change_7d),
        "avg_vol_24h": float(avg_vol_24h), "avg_vol_7d": float(avg_vol_7d),
        "nu": float(engine.nu),
    }

@st.cache_data(ttl=3600)
def get_backtest_metrics():
    fp = "backtest_results.jsonl"
    if not os.path.exists(fp): return None
    preds = [json.loads(l.strip()) for l in open(fp) if l.strip()]
    if not preds: return None
    m = evaluate_predictions(preds)
    m["predictions_data"] = preds
    return m


# ── Plotly Helpers ───────────────────────────────────────
def theme_colors():
    if is_dark:
        return dict(bg="#1f2228", grid="rgba(255,255,255,0.04)",
            text="rgba(255,255,255,0.4)", line="#fff",
            band="rgba(255,255,255,0.07)", green="#22c55e", red="#ef4444",
            bar_pos="rgba(255,255,255,0.6)", bar_neg="rgba(255,255,255,0.2)")
    else:
        return dict(bg="#fafafa", grid="rgba(0,0,0,0.04)",
            text="rgba(0,0,0,0.4)", line="#1f2228",
            band="rgba(0,0,0,0.06)", green="#16a34a", red="#dc2626",
            bar_pos="rgba(0,0,0,0.6)", bar_neg="rgba(0,0,0,0.15)")

def base_layout(tc, height=380):
    return dict(
        paper_bgcolor=tc["bg"], plot_bgcolor=tc["bg"], height=height,
        margin=dict(l=50, r=20, t=10, b=35),
        font=dict(family="JetBrains Mono, monospace", size=10, color=tc["text"]),
        xaxis=dict(gridcolor=tc["grid"], zeroline=False, showline=True,
            linecolor=tc["grid"], linewidth=1),
        yaxis=dict(gridcolor=tc["grid"], zeroline=False, showline=True,
            linecolor=tc["grid"], linewidth=1),
    )


# ── Main App ─────────────────────────────────────────────
def main():
    tc = theme_colors()
    now_utc = datetime.now(timezone.utc)
    next_h = now_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    mins_left = int((next_h - now_utc).total_seconds() // 60)

    # ── Nav Bar ──────────────────────────────────────────
    st.markdown(f"""
    <div class="nav-bar">
        <div class="nav-brand">BTC FORECAST</div>
        <div class="nav-right">
            <div class="nav-tag"><span class="live-dot"></span> LIVE</div>
            <div class="nav-tag">NEXT BAR: {mins_left}M</div>
            <div class="nav-tag">BTCUSDT · 1H · BINANCE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.checkbox(f"{'◐' if is_dark else '◑'} {'DARK' if is_dark else 'LIGHT'} MODE",
        value=not is_dark, on_change=toggle_theme, key="theme_toggle")

    # ── Load Data ────────────────────────────────────────
    backtest = get_backtest_metrics()
    try:
        with st.spinner(""):
            d = get_live_data()
    except Exception as e:
        st.error(f"Model error: {e}"); return

    current = d["current_price"]
    low, high = d["predicted_low"], d["predicted_high"]
    width = high - low
    info = d["model_info"]
    regime = info.get("volatility_regime", "").replace("🟢 ", "").replace("🟡 ", "").replace("🔴 ", "")
    chg_sign = "+" if d["change_24h"] >= 0 else ""

    # ── Hero + 24h Stats ─────────────────────────────────
    st.markdown(f"""
    <div class="hero-section">
        <div class="hero-label">BITCOIN / USDT</div>
        <div class="hero-price">${current:,.2f}</div>
        <div class="hero-subtitle">
            24h: {chg_sign}{d['change_24h']:.2f}% &nbsp;·&nbsp;
            H: ${d['high_24h']:,.2f} &nbsp;·&nbsp;
            L: ${d['low_24h']:,.2f} &nbsp;·&nbsp;
            Vol: {d['avg_vol_24h']:.4f} &nbsp;·&nbsp;
            {regime}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Metrics Grid ─────────────────────────────────────
    if backtest:
        fm = format_metrics_display(backtest)
        cov_raw = backtest.get("coverage_95", 0)
        cov_st = "ON TARGET" if 0.93 <= cov_raw <= 0.97 else "REVIEW"
        st.markdown(f"""
        <div class="metrics-grid">
            <div class="metric-card"><div class="metric-label">COVERAGE</div>
                <div class="metric-value">{fm['coverage']}</div>
                <div class="metric-sub">{cov_st} · TARGET 95%</div></div>
            <div class="metric-card"><div class="metric-label">AVG WIDTH</div>
                <div class="metric-value">{fm['avg_width']}</div>
                <div class="metric-sub">PREDICTION RANGE</div></div>
            <div class="metric-card"><div class="metric-label">WINKLER</div>
                <div class="metric-value">{fm['winkler']}</div>
                <div class="metric-sub">LOWER IS BETTER</div></div>
            <div class="metric-card"><div class="metric-label">PREDICTIONS</div>
                <div class="metric-value">{fm['predictions']}</div>
                <div class="metric-sub">{fm['hits']} HITS · {fm['misses']} MISSES</div></div>
        </div>
        """, unsafe_allow_html=True)

    # ── Prediction Range ─────────────────────────────────
    st.markdown(f"""
    <div class="prediction-card">
        <div class="pred-left">
            <div class="pred-range-label">95% CONFIDENCE INTERVAL — NEXT HOUR</div>
            <div class="pred-range-value">${low:,.2f} &nbsp;—&nbsp; ${high:,.2f}</div>
        </div>
        <div class="pred-right">
            <div class="pred-detail"><div class="pred-detail-label">WIDTH</div>
                <div class="pred-detail-value">${width:,.2f}</div></div>
            <div class="pred-detail"><div class="pred-detail-label">DRIFT</div>
                <div class="pred-detail-value">{info.get('hourly_drift',0):.6f}</div></div>
            <div class="pred-detail"><div class="pred-detail-label">SIMS</div>
                <div class="pred-detail-value">10,000</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Main Chart ───────────────────────────────────────
    st.markdown("""<div class="chart-section"><div class="chart-header">
        <div class="chart-title">Price — Last 50 Bars with 95% Confidence Ribbon</div>
        <div class="chart-tag">FIGARCH(1,0,1) + STUDENT-T</div>
    </div></div>""", unsafe_allow_html=True)

    dates = d["chart_dates"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=d["chart_highs"], mode='lines',
        line=dict(width=0), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=d["chart_lows"], mode='lines',
        line=dict(width=0), fill='tonexty', fillcolor=tc["band"],
        name='95% CI', hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=dates, y=d["chart_prices"], mode='lines',
        line=dict(color=tc["line"], width=1.5), name='BTC/USDT',
        hovertemplate='%{x|%b %d %H:%M}<br>$%{y:,.2f}<extra></extra>'))
    if dates:
        last = dates[-1]; nxt = last + pd.Timedelta(hours=1)
        fig.add_trace(go.Scatter(x=[last,nxt,nxt,last], y=[current,high,low,current],
            fill='toself', fillcolor="rgba(34,197,94,0.12)",
            line=dict(color=tc["green"], width=1),
            name=f'Pred: ${low:,.0f}–${high:,.0f}', hoverinfo='skip'))
        fig.add_hline(y=high, line_dash="dot", line_color=tc["green"], line_width=1,
            annotation_text=f"${high:,.0f}", annotation_font_color=tc["green"],
            annotation_font_size=10, annotation_font_family="JetBrains Mono")
        fig.add_hline(y=low, line_dash="dot", line_color=tc["green"], line_width=1,
            annotation_text=f"${low:,.0f}", annotation_font_color=tc["green"],
            annotation_font_size=10, annotation_font_family="JetBrains Mono")
    layout = base_layout(tc, 400)
    layout["yaxis"]["tickformat"] = "$,.0f"
    layout["legend"] = dict(orientation="h", y=1.06, x=0, font=dict(size=9, color=tc["text"]))
    layout["hovermode"] = "x unified"
    fig.update_layout(**layout)
    st.plotly_chart(fig, width='stretch')

    # ═══════════════════════════════════════════════════════
    # ADDITIONAL ANALYTICS PANELS (3-column grid)
    # ═══════════════════════════════════════════════════════
    col1, col2, col3 = st.columns(3)

    # ── Panel 1: Hourly Returns (last 24h bar chart) ─────
    with col1:
        st.markdown("""<div class="chart-section"><div class="chart-header">
            <div class="chart-title">Hourly Returns — Last 24h</div>
            <div class="chart-tag">LOG RETURNS</div>
        </div></div>""", unsafe_allow_html=True)
        rets = d["recent_ret"]
        colors = [tc["bar_pos"] if r >= 0 else tc["bar_neg"] for r in rets]
        fig_r = go.Figure(go.Bar(
            x=d["recent_ret_dates"], y=[r*100 for r in rets],
            marker_color=colors, hovertemplate='%{x}<br>%{y:.3f}%<extra></extra>'))
        lr = base_layout(tc, 240)
        lr["yaxis"]["ticksuffix"] = "%"
        fig_r.update_layout(**lr)
        st.plotly_chart(fig_r, width='stretch')

    # ── Panel 2: Volatility Timeline ─────────────────────
    with col2:
        st.markdown("""<div class="chart-section"><div class="chart-header">
            <div class="chart-title">Conditional Volatility σ</div>
            <div class="chart-tag">FIGARCH</div>
        </div></div>""", unsafe_allow_html=True)
        fig_v = go.Figure(go.Scatter(
            x=d["vol_dates"], y=d["vol_series"], mode='lines',
            line=dict(color=tc["line"], width=1), fill='tozeroy',
            fillcolor=tc["band"],
            hovertemplate='%{x|%b %d %H:%M}<br>σ = %{y:.6f}<extra></extra>'))
        fig_v.update_layout(**base_layout(tc, 240))
        st.plotly_chart(fig_v, width='stretch')

    # ── Panel 3: Monte Carlo Distribution ────────────────
    with col3:
        st.markdown("""<div class="chart-section"><div class="chart-header">
            <div class="chart-title">Simulated Next-Hour Prices</div>
            <div class="chart-tag">MC DISTRIBUTION</div>
        </div></div>""", unsafe_allow_html=True)
        sims = d["sims_terminal"]
        fig_h = go.Figure(go.Histogram(
            x=sims, nbinsx=60, marker_color=tc["bar_pos"],
            marker_line_width=0, opacity=0.8,
            hovertemplate='$%{x:,.0f}<br>Count: %{y}<extra></extra>'))
        fig_h.add_vline(x=current, line_dash="solid", line_color=tc["green"], line_width=1,
            annotation_text="Current", annotation_font_color=tc["green"],
            annotation_font_size=9, annotation_font_family="JetBrains Mono")
        hl = base_layout(tc, 240)
        hl["xaxis"]["tickformat"] = "$,.0f"
        fig_h.update_layout(**hl)
        st.plotly_chart(fig_h, width='stretch')

    # ═══════════════════════════════════════════════════════
    # MARKET STATS + MODEL PARAMS + HISTORY (3-panel bottom)
    # ═══════════════════════════════════════════════════════
    def stat_row(k, v):
        return f'<div class="param-row"><span class="param-key">{k}</span><span class="param-val">{v}</span></div>'

    # Market stats card (HTML)
    stats_html = f"""
    <div class="stats-grid" style="display:grid;grid-template-columns:1fr 1fr 1.5fr;gap:1px;
        border:1px solid {'rgba(255,255,255,0.1)' if is_dark else 'rgba(0,0,0,0.1)'};
        margin-bottom:48px;animation:fadeInUp 0.7s ease-out 0.5s both;">
        <div class="bottom-panel">
            <div class="panel-title">MARKET STATISTICS</div>
            {stat_row("24h Change", f'{chg_sign}{d["change_24h"]:.2f}%')}
            {stat_row("24h High", f'${d["high_24h"]:,.2f}')}
            {stat_row("24h Low", f'${d["low_24h"]:,.2f}')}
            {stat_row("24h Volatility", f'{d["avg_vol_24h"]:.6f}')}
            {stat_row("7d Change", f'{"+" if d["change_7d"]>=0 else ""}{d["change_7d"]:.2f}%')}
            {stat_row("7d High", f'${d["high_7d"]:,.2f}')}
            {stat_row("7d Low", f'${d["low_7d"]:,.2f}')}
            {stat_row("7d Volatility", f'{d["avg_vol_7d"]:.6f}')}
        </div>
        <div class="bottom-panel">
            <div class="panel-title">MODEL PARAMETERS</div>
            {stat_row("Hourly Drift (μ)", f'{info.get("hourly_drift",0):.8f}')}
            {stat_row("Student-t ν", f'{d["nu"]:.2f}')}
            {stat_row("Latest σ", f'{info.get("latest_sigma",0):.6f}')}
            {stat_row("Mean σ²", f'{info.get("mean_sigma2",0):.10f}')}
            {stat_row("Regime", regime.upper())}
            {stat_row("Simulations", "10,000")}
            {stat_row("Train Window", "500 BARS")}
            {stat_row("Time Step (dt)", "1.0 (HOURLY)")}
        </div>
        <div class="bottom-panel">
            <div class="panel-title">PREDICTION HISTORY</div>
        </div>
    </div>"""
    st.markdown(stats_html, unsafe_allow_html=True)

    # History table
    store = PredictionStore(use_gsheets=True, sheet_url="https://docs.google.com/spreadsheets/d/1nnQH3URcdwTImRJo54TawKC23MD6fcLOnKucVTgHOtk/edit?gid=0#gid=0")
    
    # Verify past predictions using the last 50 closed bars
    price_map = {dt.isoformat(): p for dt, p in zip(d["chart_dates"], d["chart_prices"])}
    store.verify_predictions(price_map)

    if store.should_save_new_prediction():
        store.save_prediction({
            "timestamp": d["prediction_time"], "current_price": current,
            "predicted_low_95": low, "predicted_high_95": high,
        })
    hdf = store.get_history_dataframe()
    if len(hdf) > 0:
        st.dataframe(hdf, width='stretch', hide_index=True)
    else:
        st.markdown(f"""<p style="font-family:Inter,sans-serif;font-size:13px;
            color:{'rgba(255,255,255,0.3)' if is_dark else 'rgba(0,0,0,0.3)'};
            padding:8px 0;">No predictions stored yet. Each visit saves a new entry.</p>""",
            unsafe_allow_html=True)

    # ── Footer ───────────────────────────────────────────
    st.markdown(f"""
    <div class="site-footer">
        <div class="footer-text">AlphaI × Polaris Challenge &nbsp;·&nbsp;
            GBM + FIGARCH(1,0,1) + Student-t &nbsp;·&nbsp; Binance BTCUSDT 1H</div>
        <div class="footer-mono">{now_utc.strftime('%Y-%m-%d %H:%M:%S UTC')}</div>
    </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
