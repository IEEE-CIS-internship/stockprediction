import os
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="AuraTrade AI Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply premium styling
st.markdown("""
<style>
    .reportview-container {
        background-color: #0f111a;
    }
    .main {
        background-color: #0f111a;
        color: white;
    }
    div.stButton > button:first-child {
        background-color: #17becf;
        color: white;
        border-radius: 6px;
    }
    .metric-card {
        background-color: #151722;
        padding: 18px;
        border-radius: 8px;
        border: 1px solid #2e3039;
        margin-bottom: 12px;
    }
    .recommendation-card {
        background-color: #182232;
        padding: 20px;
        border-radius: 8px;
        border: 1px solid #1f3554;
        margin-bottom: 20px;
        font-size: 1.05rem;
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# Configure Matplotlib styles
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "figure.facecolor": "#0f111a",
    "axes.facecolor": "#151722",
    "savefig.facecolor": "#0f111a",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#2e3039",
    "axes.edgecolor": "#2e3039",
    "font.size": 9
})

API_URL = "http://127.0.0.1:8000"

st.title("⚡ AuraTrade AI: Decoupled Swing Trading Intelligence")
st.caption("Gaussian HMM Regime-Gated Mixture-of-Experts (MoE) Forecasting System")

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.header("🛠️ System Controls")

# Fetch available tickers
try:
    tickers_resp = requests.get(f"{API_URL}/api/tickers").json()
    tickers_list = tickers_resp["tickers"]
except Exception:
    tickers_list = ["ICICIBANK.NS", "INDIGO.NS", "MARUTI.NS", "TRENT.NS", "HCLTECH.NS", "HINDALCO.NS", "ONGC.NS", "ADANIENT.NS"]
    st.sidebar.error("Could not connect to FastAPI backend server. Ensure backend is running at http://127.0.0.1:8000.")

ticker = st.sidebar.selectbox("Select Asset Ticker", tickers_list, index=0)

horizon_map = {"1-Day Swing": 1, "3-Day Swing": 3, "7-Day Swing": 7}
horizon_label = st.sidebar.selectbox("Forecast Horizon", list(horizon_map.keys()), index=1)
horizon = horizon_map[horizon_label]

lookback_days = st.sidebar.slider("Historical Data Points", min_value=50, max_value=300, value=120)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**System Architecture:**
1. **Exogenous Ingest**: Brent crude, USD/INR, News RSS RSS headlines.
2. **Dynamic Gating**: 3-State Gaussian HMM (Bullish, Bearish, Sideways).
3. **MoE Forecast**: Shared LSTM backbone routing outputs to expert networks.
4. **Attribution**: Gradient-based saliency mappings.
""")

# -----------------------------------------------------------------------------
# Fetch Data from REST endpoints
# -----------------------------------------------------------------------------
@st.cache_data(ttl=10)
def fetch_api_data(ticker_val, horizon_val, limit_val):
    try:
        data_resp = requests.get(f"{API_URL}/api/data/{ticker_val}?limit={limit_val}").json()
        regime_resp = requests.get(f"{API_URL}/api/regime/{ticker_val}").json()
        forecast_resp = requests.get(f"{API_URL}/api/forecast/{ticker_val}?horizon={horizon_val}").json()
        explain_resp = requests.get(f"{API_URL}/api/explain/{ticker_val}?horizon={horizon_val}").json()
        backtest_resp = requests.get(f"{API_URL}/api/backtest/{ticker_val}?horizon={horizon_val}").json()
        return data_resp, regime_resp, forecast_resp, explain_resp, backtest_resp
    except Exception as e:
        return None, None, None, None, None

data_payload, regime_payload, forecast_payload, explain_payload, backtest_payload = fetch_api_data(ticker, horizon, lookback_days)

if data_payload is None:
    st.warning("FastAPI backend is offline. Start the server using: `uvicorn backend.app:app`")
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
        
        regime_colors = {"Bullish": "#2ca02c", "Bearish": "#d62728", "Sideways": "#ff7f0e"}
        color = regime_colors.get(regime, "white")
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0;color:#7f7f7f;">Market Regime State</h4>
            <h2 style="margin:5px 0;color:{color};">{regime}</h2>
            <p style="margin:0;font-size:0.85rem;color:#7f7f7f;">Gated via HMM Posterior Probabilities</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        # Forecast Return Card
        pred_ret = forecast_payload["predicted_residual_return"]
        buy_thresh = forecast_payload["volatility_threshold_buy"]
        sell_thresh = forecast_payload["volatility_threshold_sell"]
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0;color:#7f7f7f;">Predicted Residual Return</h4>
            <h2 style="margin:5px 0;color:#17becf;">{pred_ret * 100:+.3f}%</h2>
            <p style="margin:0;font-size:0.85rem;color:#7f7f7f;">Buy: &gt; {buy_thresh*100:.2f}% | Sell: &lt; {sell_thresh*100:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        # Trading Signal Card
        signal = forecast_payload["trading_signal"]
        signal_colors = {"BUY": "#2ca02c", "SELL": "#d62728", "HOLD": "#7f7f7f"}
        sig_color = signal_colors.get(signal, "white")
        
        st.markdown(f"""
        <div class="metric-card">
            <h4 style="margin:0;color:#7f7f7f;">AuraTrade Signal</h4>
            <h2 style="margin:5px 0;color:{sig_color};">{signal}</h2>
            <p style="margin:0;font-size:0.85rem;color:#7f7f7f;">Confidence Gate Ablated &amp; Applied</p>
        </div>
        """, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Row 2: Charts
    # -------------------------------------------------------------------------
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
        
        ax.plot(dates, prices, color="#17becf", label=f"{ticker} Close", linewidth=2.0)
        
        # Shading regions
        # Bullish (0) -> Green, Bearish (1) -> Red, Sideways (2) -> Orange
        # Shade each daily segment
        if "regime" in df_plot.columns:
            df_plot["regime"] = df_plot["regime"].fillna(2)
            reg_vals = df_plot["regime"].values
            for i in range(len(dates) - 1):
                state = int(reg_vals[i])
                if state == 0:
                    ax.axvspan(dates[i], dates[i+1], color="green", alpha=0.15)
                elif state == 1:
                    ax.axvspan(dates[i], dates[i+1], color="red", alpha=0.15)
                else:
                    ax.axvspan(dates[i], dates[i+1], color="orange", alpha=0.15)
                    
        ax.set_ylabel("Price (INR)", color="white")
        ax.set_title(f"Regime Timeline Segmentation Chart ({ticker})", color="white", fontsize=11)
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
        colors = ["#2ca02c" if val >= 0 else "#d62728" for val in df_attr["Attribution (%)"]]
        sns.barplot(data=df_attr, x="Attribution (%)", y="Feature", palette=colors, hue="Feature", legend=False, ax=ax)
        ax.set_xlabel("Attribution Contribution (%)", color="white")
        ax.set_ylabel("Input Features", color="white")
        ax.set_title("Neural Feature Saliency (Top 8 Drivers)", color="white", fontsize=11)
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
            st.dataframe(df_compare, hide_index=True, use_container_width=True)
            
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
            st.image(fig_path, use_column_width=True)
        else:
            st.info("Backtest equity curve chart not found.")
            
# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.caption("AuraTrade AI System • IEEE-CIS Internship Stock Prediction Project • Version 1.0.0")
