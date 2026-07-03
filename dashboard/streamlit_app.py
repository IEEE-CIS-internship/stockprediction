import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

THEMES = {
    "dark": {
        "accent_soft": "#5eead4",
        "bullish": "#3dd68c",
        "bearish": "#f87171",
        "sideways": "#fbbf24",
        "hold": "#94a3b8",
        "chart_bg": "#161b28",
        "grid": "#252b3d",
        "border": "#2d3650",
        "text": "#f0f2f8",
        "muted": "#8b93a8",
        "bg": "#0f111a",
        "regime_alpha": 0.18,
    },
    "light": {
        "accent_soft": "#0e7490",
        "bullish": "#059669",
        "bearish": "#dc2626",
        "sideways": "#d97706",
        "hold": "#64748b",
        "chart_bg": "#ffffff",
        "grid": "#e2e8f0",
        "border": "#e2e8f0",
        "text": "#0b1c30",
        "muted": "#64748b",
        "bg": "#f5f7fb",
        "regime_alpha": 0.22,
    },
}

THEME_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@500;600;700&display=swap');

    .dashboard-title {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        line-height: 1.25;
        margin: 0 0 0.35rem 0;
        padding: 0;
    }
    .dashboard-title .brand {
        color: var(--primary-color);
        font-weight: 700;
    }
    .dashboard-title .tagline {
        color: var(--text-color);
        font-weight: 600;
    }
    .dashboard-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1rem;
        font-weight: 500;
        color: var(--text-color);
        opacity: 0.65;
        margin: 0 0 1.25rem 0;
    }

    .metric-card {
        background: var(--secondary-background-color);
        padding: 18px 20px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }
    .metric-card.forecast {
        border-left-color: var(--primary-color);
    }
    .metric-label {
        margin: 0;
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: var(--text-color);
        opacity: 0.6;
    }
    .metric-value {
        margin: 8px 0 6px;
        font-family: 'Inter', sans-serif;
        font-size: 1.65rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .metric-foot {
        margin: 0;
        font-size: 0.85rem;
        color: var(--text-color);
        opacity: 0.5;
    }

    .recommendation-card {
        background: var(--secondary-background-color);
        padding: 20px 22px;
        border-radius: 10px;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 20px;
        font-size: 1.05rem;
        line-height: 1.65;
        color: var(--text-color);
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.08);
    }

    h3 {
        color: var(--text-color) !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
    }

    div[data-testid="stDataFrame"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        overflow: hidden;
    }
</style>
"""


def theme_base() -> str:
    ctx = st.context.theme
    base = ctx.get("base") if hasattr(ctx, "get") else getattr(ctx, "base", None)
    if base in ("light", "dark"):
        return base
    return st.get_option("theme.base") or "dark"


def chart_palette() -> dict:
    palette = THEMES[theme_base()].copy()
    ctx = st.context.theme
    if hasattr(ctx, "get"):
        if ctx.get("backgroundColor"):
            palette["bg"] = ctx["backgroundColor"]
        if ctx.get("secondaryBackgroundColor"):
            palette["chart_bg"] = ctx["secondaryBackgroundColor"]
        if ctx.get("textColor"):
            palette["text"] = ctx["textColor"]
    return palette


def apply_mpl_theme(palette: dict) -> None:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    plt.rcParams.update({
        "figure.facecolor": palette["chart_bg"],
        "axes.facecolor": palette["chart_bg"],
        "savefig.facecolor": palette["chart_bg"],
        "text.color": palette["text"],
        "axes.labelcolor": palette["muted"],
        "xtick.color": palette["muted"],
        "ytick.color": palette["muted"],
        "grid.color": palette["grid"],
        "axes.edgecolor": palette["border"],
        "font.size": 9,
    })


def finalize_chart(fig, ax, palette: dict) -> None:
    fig.patch.set_facecolor(palette["chart_bg"])
    ax.set_facecolor(palette["chart_bg"])
    for spine in ax.spines.values():
        spine.set_color(palette["border"])

# Configure page layout
st.set_page_config(
    page_title="AuraTrade AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_URL = "http://127.0.0.1:8000"

palette = chart_palette()
st.markdown(THEME_CSS, unsafe_allow_html=True)
apply_mpl_theme(palette)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("🛠️ System Controls")
st.markdown(
    """
    <h1 class="dashboard-title">
        <span class="brand">AuraTrade AI</span><span class="tagline">: Decoupled Swing Trading Intelligence</span>
    </h1>
    <p class="dashboard-subtitle">Gaussian HMM Regime-Gated Mixture-of-Experts (MoE) Forecasting System</p>
    """,
    unsafe_allow_html=True,
)

# Fetch available tickers
try:
    tickers_resp = requests.get(f"{API_URL}/api/tickers", timeout=8).json()
    tickers_list = tickers_resp["tickers"]
    if not tickers_list:
        tickers_list = ["ICICIBANK.NS", "INDIGO.NS", "MARUTI.NS", "TRENT.NS", "HCLTECH.NS", "HINDALCO.NS", "ONGC.NS", "ADANIENT.NS"]
        st.sidebar.warning("No processed feature files found on the backend yet.")
except Exception:
    tickers_list = ["ICICIBANK.NS", "INDIGO.NS", "MARUTI.NS", "TRENT.NS", "HCLTECH.NS", "HINDALCO.NS", "ONGC.NS", "ADANIENT.NS"]
    st.sidebar.error("Could not connect to FastAPI backend server. Ensure backend is running at http://127.0.0.1:8000.")

ticker = st.sidebar.selectbox("Select Asset Ticker", tickers_list, index=0)

horizon_map = {"1-Day Swing": 1, "3-Day Swing": 3, "7-Day Swing": 7}
horizon_label = st.sidebar.selectbox("Forecast Horizon", list(horizon_map.keys()), index=1)
horizon = horizon_map[horizon_label]

lookback_days = st.sidebar.slider("Historical Data Points", min_value=50, max_value=300, value=120)

if st.sidebar.button("Refresh Selected Data"):
    with st.spinner(f"Refreshing {ticker} from yFinance..."):
        try:
            refresh_resp = requests.post(f"{API_URL}/api/refresh/{ticker}", timeout=300)
            if refresh_resp.ok:
                st.cache_data.clear()
                st.sidebar.success(f"{ticker} refreshed.")
                st.rerun()
            else:
                detail = refresh_resp.json().get("detail", refresh_resp.text)
                st.sidebar.error(f"Refresh failed: {detail}")
        except Exception as exc:
            st.sidebar.error(f"Refresh failed: {exc}")

if st.sidebar.button("Refresh All Data"):
    with st.spinner("Refreshing all project tickers from yFinance..."):
        try:
            refresh_resp = requests.post(f"{API_URL}/api/refresh/all", timeout=600)
            if refresh_resp.ok:
                st.cache_data.clear()
                st.sidebar.success("All tickers refreshed.")
                st.rerun()
            else:
                detail = refresh_resp.json().get("detail", refresh_resp.text)
                st.sidebar.error(f"Refresh failed: {detail}")
        except Exception as exc:
            st.sidebar.error(f"Refresh failed: {exc}")

# -----------------------------------------------------------------------------
# Fetch Data from REST endpoints
# -----------------------------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_api_data(ticker_val, horizon_val, limit_val):
    def api_get(path):
        response = requests.get(f"{API_URL}{path}", timeout=15)
        payload = response.json()
        if not response.ok:
            detail = payload.get("detail", response.text) if isinstance(payload, dict) else response.text
            return {"error": detail, "status_code": response.status_code}
        return payload

    try:
        data_resp = api_get(f"/api/data/{ticker_val}?limit={limit_val}")
        regime_resp = api_get(f"/api/regime/{ticker_val}")
        forecast_resp = api_get(f"/api/forecast/{ticker_val}?horizon={horizon_val}")
        explain_resp = api_get(f"/api/explain/{ticker_val}?horizon={horizon_val}")
        backtest_resp = api_get(f"/api/backtest/{ticker_val}?horizon={horizon_val}")
        return data_resp, regime_resp, forecast_resp, explain_resp, backtest_resp
    except Exception:
        return None, None, None, None, None

data_payload, regime_payload, forecast_payload, explain_payload, backtest_payload = fetch_api_data(ticker, horizon, lookback_days)

if data_payload is None:
    st.warning("FastAPI backend is offline. Start the server using: `uvicorn backend.app:app`")
elif any(isinstance(payload, dict) and "error" in payload for payload in [data_payload, regime_payload, forecast_payload, explain_payload, backtest_payload]):
    st.error("Dashboard data is not available for the selected ticker yet.")
    for label, payload in [
        ("Market data", data_payload),
        ("Regime", regime_payload),
        ("Forecast", forecast_payload),
        ("Explainability", explain_payload),
        ("Backtest", backtest_payload),
    ]:
        if isinstance(payload, dict) and "error" in payload:
            st.info(f"{label}: {payload['error']}")
else:
    # -------------------------------------------------------------------------
    # Row 1: KPI Panels
    # -------------------------------------------------------------------------
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Active Regime Card
        regime = regime_payload["active_regime"]
        regime_idx = regime_payload["regime_index"]
        probs = regime_payload["probabilities"]
        
        regime_colors = {"Bullish": palette["bullish"], "Bearish": palette["bearish"], "Sideways": palette["sideways"]}
        color = regime_colors.get(regime, palette["text"])        
        st.markdown(f"""
        <div class="metric-card regime" style="--card-accent: {color}; border-left-color: {color};">
            <h4 class="metric-label">Market Regime State</h4>
            <h2 class="metric-value" style="color:{color};">{regime}</h2>
            <p class="metric-foot">Gated via HMM Posterior Probabilities</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Forecast Return Card
        pred_ret = forecast_payload["predicted_residual_return"]
        buy_thresh = forecast_payload["volatility_threshold_buy"]
        sell_thresh = forecast_payload["volatility_threshold_sell"]
        
        st.markdown(f"""
        <div class="metric-card forecast">
            <h4 class="metric-label">Predicted Residual Return</h4>
            <h2 class="metric-value" style="color:{palette['accent_soft']};">{pred_ret * 100:+.3f}%</h2>
            <p class="metric-foot">Buy: &gt; {buy_thresh*100:.2f}% | Sell: &lt; {sell_thresh*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Trading Signal Card
        signal = forecast_payload["trading_signal"]
        signal_colors = {"BUY": palette["bullish"], "SELL": palette["bearish"], "HOLD": palette["hold"]}
        sig_color = signal_colors.get(signal, palette["text"])        
        st.markdown(f"""
        <div class="metric-card signal" style="--card-accent: {sig_color}; border-left-color: {sig_color};">
            <h4 class="metric-label">AuraTrade Signal</h4>
            <h2 class="metric-value" style="color:{sig_color};">{signal}</h2>
            <p class="metric-foot">Confidence Gate Ablated &amp; Applied</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Row 2: Charts
    # -------------------------------------------------------------------------
    palette = chart_palette()
    apply_mpl_theme(palette)

    col_chart, col_explain = st.columns([3, 2])
    
    with col_chart:
        st.subheader("📈 Close Price timeline & Regime Segment Shading")
        
        records = data_payload["data"]
        df_plot = pd.DataFrame(records)
        df_plot["date"] = pd.to_datetime(df_plot["date"])
        
        # Load processed regimes to shade background
        # (For simpler UI, we call the local file if it exists, or estimate states from HMM probabilities)
        regimes_path = f"data/processed/regimes/{ticker}_regime.csv"
        if os.path.exists(regimes_path):
            df_reg = pd.read_csv(regimes_path)
            df_reg["date"] = pd.to_datetime(df_reg["date"])
            df_plot = pd.merge(df_plot, df_reg[["date", "regime"]], on="date", how="left")
        else:
            df_plot["regime"] = regime_idx
            
        fig, ax = plt.subplots(figsize=(10, 4.5))
        dates = df_plot["date"].values
        prices = df_plot["close"].values
        
        regime_fill = {0: palette["bullish"], 1: palette["bearish"], 2: palette["sideways"]}

        ax.plot(dates, prices, color=palette["accent_soft"], label=f"{ticker} Close", linewidth=2.2)
        
        if "regime" in df_plot.columns:
            df_plot["regime"] = df_plot["regime"].fillna(2)
            reg_vals = df_plot["regime"].values
            for i in range(len(dates) - 1):
                state = int(reg_vals[i])
                ax.axvspan(
                    dates[i], dates[i + 1],
                    color=regime_fill.get(state, palette["sideways"]),
                    alpha=palette["regime_alpha"],
                )
                    
        ax.set_ylabel("Price (INR)", color=palette["muted"])
        ax.set_title(f"Regime Timeline Segmentation Chart ({ticker})", color=palette["text"], fontsize=11, pad=10)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        finalize_chart(fig, ax, palette)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
    with col_explain:
        st.subheader("🧠 Explainability Attribution Drivers")
        
        attributions = explain_payload["feature_attributions"]
        df_attr = pd.DataFrame(list(attributions.items()), columns=["Feature", "Attribution (%)"])
        df_attr["abs_attr"] = df_attr["Attribution (%)"].abs()
        df_attr = df_attr.sort_values("abs_attr", ascending=False).head(8)
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        # Color bar based on sign of attribution
        colors = [palette["bullish"] if val >= 0 else palette["bearish"] for val in df_attr["Attribution (%)"]]
        sns.barplot(data=df_attr, x="Attribution (%)", y="Feature", palette=colors, hue="Feature", legend=False, ax=ax)
        ax.set_xlabel("Attribution Contribution (%)", color=palette["muted"])
        ax.set_ylabel("Input Features", color=palette["muted"])
        ax.set_title("Neural Feature Saliency (Top 8 Drivers)", color=palette["text"], fontsize=11, pad=10)
        ax.grid(True, axis="x", linestyle="--", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        finalize_chart(fig, ax, palette)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # -------------------------------------------------------------------------
    # Row 3: Natural Language Reasoning & Recommendations
    # -------------------------------------------------------------------------
    st.subheader("📝 Natural Language Reasoning Recommendation")
    reasoning_text = explain_payload["recommendation_reasoning"]
    st.markdown(f"""
    <div class="recommendation-card">
        {reasoning_text}
    </div>
    """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Row 4: Performance & Backtesting
    # -------------------------------------------------------------------------
    st.subheader("📊 Backtesting Scorecard & Gating Ablation Study")
    
    col_bt_metrics, col_bt_curve = st.columns([2, 3])
    
    with col_bt_metrics:
        gated_record = backtest_payload["gated_configuration"]
        ungated_record = backtest_payload["ungated_configuration"]
        
        if gated_record and ungated_record:
            df_compare = pd.DataFrame([
                {
                    "Metric": "Sharpe Ratio",
                    "Gated MoE (Proposed)": f"{gated_record['Sharpe']:.2f}",
                    "Ungated MoE (Ablated)": f"{ungated_record['Sharpe']:.2f}"
                },
                {
                    "Metric": "Max Drawdown (%)",
                    "Gated MoE (Proposed)": f"{gated_record['Max Drawdown']*100:.2f}%",
                    "Ungated MoE (Ablated)": f"{ungated_record['Max Drawdown']*100:.2f}%"
                },
                {
                    "Metric": "Prediction Hit Rate (%)",
                    "Gated MoE (Proposed)": f"{gated_record['Hit Rate (%)']:.1f}%",
                    "Ungated MoE (Ablated)": f"{ungated_record['Hit Rate (%)']:.1f}%"
                },
                {
                    "Metric": "Total Trades Executed",
                    "Gated MoE (Proposed)": int(gated_record["Signals"]),
                    "Ungated MoE (Ablated)": int(ungated_record["Signals"])
                }
            ])
            st.dataframe(df_compare, hide_index=True, width="stretch")
            
            st.markdown("""
            > [!TIP]
            > **Confidence Gating Benefit**: The Gated MoE applies HMM probabilities to filter out low-confidence signals and prevent trading during high-uncertainty regime transitions, leading to improved Sharpe ratios and reduced drawdown.
            """)
        else:
            st.info("No backtest performance metrics available on the server.")
            
    with col_bt_curve:
        # Load pre-saved backtest equity curves png
        fig_path = f"outputs/figures/{ticker}_backtest_equity_h{horizon}.png"
        if os.path.exists(fig_path):
            st.image(fig_path, width="stretch")
        else:
            st.info("Backtest equity curve chart not found.")
            
# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("AuraTrade AI System • IEEE-CIS Internship Stock Prediction Project • Version 1.0.0")
