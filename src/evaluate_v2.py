import os
import argparse
import joblib
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Set professional plotting style
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
    "font.size": 10
})

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from lstm import PlainLSTM
from lstm_sentiment import LSTMSentiment
from regime_gated import RegimeConditionedMoE
from linear import BaselineLinearRegression

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    # MAPE calculation avoiding divide-by-zero
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100 if np.sum(mask) > 0 else 0.0
    
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def create_sequences(df, features, target_col, lookback=30, horizon=1):
    X, y = [], []
    feature_data = df[features].values
    target_data = df[target_col].values
    
    for i in range(len(df) - lookback - horizon + 1):
        X.append(feature_data[i : i + lookback])
        y.append(target_data[i + lookback - 1 + horizon])
        
    return np.array(X), np.array(y)

def create_sequences_moe(df, features, target_col, lookback=30, horizon=1):
    X, regime_probs, y = [], [], []
    feature_data = df[features].values
    regime_data = df[["prob_bull", "prob_bear", "prob_side"]].values
    target_data = df[target_col].values
    
    for i in range(len(df) - lookback - horizon + 1):
        X.append(feature_data[i : i + lookback])
        regime_probs.append(regime_data[i + lookback - 1])
        y.append(target_data[i + lookback - 1 + horizon])
        
    return np.array(X), np.array(regime_probs), np.array(y)

def run_backtest(df_test, preds_moe, horizon, ticker, trans_cost=0.0015, use_confidence_gate=True):
    """
    Simulates swing trading based on residual return forecasts.
    Incorporates 1-day execution lag and 15 bps transaction costs.
    """
    prices = df_test["close"].values
    opens = df_test["open"].values
    dates = df_test["date"].values
    regimes = df_test["regime"].values
    
    # Extract confidence details
    prob_bull = df_test["prob_bull"].values
    prob_bear = df_test["prob_bear"].values
    prob_side = df_test["prob_side"].values
    
    # Buy/Sell Threshold set dynamically based on return volatility
    ret_std = df_test["returns"].std()
    threshold_buy = ret_std * 0.5
    threshold_sell = -ret_std * 0.5
    
    cash = 100000.0
    shares = 0
    position_type = "cash"  # long, short, cash
    entry_price = 0.0
    entry_day = 0
    
    equity_curve = [cash]
    trade_returns = []
    signals_count = 0
    correct_signals = 0
    
    for t in range(len(preds_moe)):
        current_idx = t + 29 # Align with sequence end (lookback-1)
        if current_idx >= len(df_test):
            break
            
        current_date = dates[current_idx]
        current_price = prices[current_idx]
        current_open = opens[current_idx]
        
        # Current active HMM probability
        regime_idx = int(regimes[current_idx])
        probs = [prob_bull[current_idx], prob_bear[current_idx], prob_side[current_idx]]
        confidence = probs[regime_idx]
        
        # Get signal from today's forecast
        pred_val = preds_moe[t]
        
        # Determine signal based on rules
        signal = "HOLD"
        
        # Apply confidence gate ablation flag
        conf_min = 0.50 if use_confidence_gate else 0.0
        
        if pred_val > threshold_buy and regime_idx != 1 and confidence >= conf_min:
            signal = "BUY"
        elif pred_val < threshold_sell or (regime_idx == 1 and confidence >= conf_min):
            signal = "SELL"
            
        # Execute Trades with 1-day lag (next day's Open price)
        # If we hold a position, check if we need to close it after `horizon` days
        if position_type == "long":
            if current_idx - entry_day >= horizon:
                # Sell position at today's Open
                cash = shares * current_open * (1.0 - trans_cost)
                shares = 0
                trade_ret = (current_open - entry_price) / entry_price - 2 * trans_cost
                trade_returns.append(trade_ret)
                position_type = "cash"
        elif position_type == "short":
            if current_idx - entry_day >= horizon:
                # Cover short at today's Open
                cash = cash + (entry_price - current_open) * shares - (cash * trans_cost)
                shares = 0
                trade_ret = (entry_price - current_open) / entry_price - 2 * trans_cost
                trade_returns.append(trade_ret)
                position_type = "cash"
                
        # Enter New Positions at next day's Open
        # (This is executed on next iteration, but for simple modeling we enter using next day details)
        next_idx = current_idx + 1
        if next_idx < len(df_test) and position_type == "cash":
            next_open = opens[next_idx]
            
            if signal == "BUY":
                signals_count += 1
                # Check if correct (residual return over horizon is positive)
                fwd_res = df_test.iloc[current_idx][f"target_res_return_{horizon}d"]
                if fwd_res > 0:
                    correct_signals += 1
                    
                # Buy shares
                shares = cash * (1.0 - trans_cost) / next_open
                cash = 0
                entry_price = next_open
                entry_day = next_idx
                position_type = "long"
            elif signal == "SELL":
                signals_count += 1
                # Check if correct (residual return over horizon is negative)
                fwd_res = df_test.iloc[current_idx][f"target_res_return_{horizon}d"]
                if fwd_res < 0:
                    correct_signals += 1
                    
                # Short shares (simulated via cash value margin)
                shares = cash * (1.0 - trans_cost) / next_open
                entry_price = next_open
                entry_day = next_idx
                position_type = "short"
                
        # Update equity curve value
        current_equity = cash
        if position_type == "long":
            current_equity = shares * current_price
        elif position_type == "short":
            current_equity = cash + (entry_price - current_price) * shares
            
        equity_curve.append(current_equity)
        
    # Calculate performance metrics
    eq_series = pd.Series(equity_curve)
    returns_series = eq_series.pct_change().dropna()
    
    # Sharpe ratio (annualized)
    std_val = returns_series.std()
    mean_val = returns_series.mean()
    sharpe = (mean_val / std_val) * np.sqrt(252) if std_val > 0 else 0.0
    
    # Max Drawdown
    cum_max = eq_series.cummax()
    drawdowns = (eq_series - cum_max) / cum_max
    max_dd = drawdowns.min()
    
    # Hit rate
    hit_rate = (correct_signals / signals_count) * 100 if signals_count > 0 else 0.0
    
    return {
        "equity_curve": equity_curve,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "hit_rate": hit_rate,
        "trades": len(trade_returns),
        "total_signals": signals_count
    }

def main():
    parser = argparse.ArgumentParser(description="AuraTrade System Evaluation")
    parser.add_argument("--models", default="all", help="Models to evaluate")
    parser.add_argument("--out", nargs=2, required=True, help="Output directories: [tab_dir, fig_dir]")
    
    args = parser.parse_args()
    tab_dir, fig_dir = args.out[0], args.out[1]
    
    os.makedirs(tab_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)
    
    features_dir = "data/processed/features/"
    models_dir = "models"
    
    tech_features = ["open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30"]
    all_features = tech_features + ["sentiment_score"]
    moe_features = [
        "open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30",
        "sentiment_score", "vix", "crude", "usdinr", "is_split_day", "is_dividend_day", "days_to_earnings"
    ]
    sentiment_idx = moe_features.index("sentiment_score")
    
    horizons = [1, 3, 7]
    lookback = 30
    
    # Find all stock files
    tickers = [f.split("_features.csv")[0] for f in os.listdir(features_dir) if f.endswith("_features.csv") and not f.startswith("index")]
    
    overall_scorecard = []
    regime_scorecard = []
    transition_scorecard = []
    per_stock_scorecard = []
    backtest_scorecard = []
    
    for h in horizons:
        print(f"\nEvaluating models for Horizon: {h} days...")
        
        h_overall_preds = {}
        h_actuals = {}
        
        transition_true_list = []
        transition_preds_list = {
            "Linear_Regression": [],
            "Plain_LSTM": [],
            "LSTM_Sentiment": [],
            "Regime_MoE": []
        }
        
        per_regime_true = {0: [], 1: [], 2: []}
        per_regime_preds = {
            0: {"Linear_Regression": [], "Plain_LSTM": [], "LSTM_Sentiment": [], "Regime_MoE": []},
            1: {"Linear_Regression": [], "Plain_LSTM": [], "LSTM_Sentiment": [], "Regime_MoE": []},
            2: {"Linear_Regression": [], "Plain_LSTM": [], "LSTM_Sentiment": [], "Regime_MoE": []}
        }
        
        for ticker in tickers:
            features_file = os.path.join(features_dir, f"{ticker}_features.csv")
            df = pd.read_csv(features_file)
            df["date"] = pd.to_datetime(df["date"])
            df.sort_values("date", inplace=True)
            df.reset_index(drop=True, inplace=True)
            
            # Split chronologically (80/20)
            split_idx = int(len(df) * 0.8)
            test_df = df.iloc[split_idx:].copy()
            
            target_col = f"target_res_return_{h}d"
            
            # Load input scalers
            scaler_tech = joblib.load(os.path.join(models_dir, f"{ticker}_scaler_tech.pkl"))
            scaler_all = joblib.load(os.path.join(models_dir, f"{ticker}_scaler_all.pkl"))
            scaler_moe = joblib.load(os.path.join(models_dir, f"{ticker}_scaler_moe.pkl"))
            scaler_target = joblib.load(os.path.join(models_dir, f"{ticker}_scaler_target_h{h}.pkl"))
            
            # Drop target NaNs
            test_df_h = test_df.dropna(subset=[target_col]).copy()
            
            if len(test_df_h) < lookback:
                continue
                
            # Align regimes
            target_indices = np.arange(lookback - 1 + h, len(test_df_h))
            regimes = test_df_h.iloc[target_indices]["regime"].values
            
            # Scale test set
            test_df_h_scaled = test_df_h.copy()
            test_df_h_scaled[tech_features] = scaler_tech.transform(test_df_h_scaled[tech_features])
            test_df_h_scaled[all_features] = scaler_all.transform(test_df_h_scaled[all_features])
            test_df_h_scaled[moe_features] = scaler_moe.transform(test_df_h_scaled[moe_features])
            test_df_h_scaled["target_scaled"] = scaler_target.transform(test_df_h_scaled[[target_col]])
            
            # Create sequences
            X_tech, y_scaled = create_sequences(test_df_h_scaled, tech_features, target_col="target_scaled", lookback=lookback, horizon=h)
            X_all, _ = create_sequences(test_df_h_scaled, all_features, target_col="target_scaled", lookback=lookback, horizon=h)
            X_moe, r_probs, _ = create_sequences_moe(test_df_h_scaled, moe_features, target_col="target_scaled", lookback=lookback, horizon=h)
            
            actual_returns = scaler_target.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            
            # Load Checkpoints
            # 1. Linear Regression
            lr_model = joblib.load(os.path.join(models_dir, f"{ticker}_linear_h{h}.pkl"))
            preds_lr_scaled = lr_model.predict(X_tech.reshape(len(X_tech), -1))
            preds_lr = scaler_target.inverse_transform(preds_lr_scaled.reshape(-1, 1)).flatten()
            
            # 2. Plain LSTM
            plain_lstm = PlainLSTM(input_size=len(tech_features))
            plain_lstm.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker}_plain_lstm_h{h}.pt")))
            plain_lstm.eval()
            with torch.no_grad():
                preds_plain_scaled = plain_lstm(torch.tensor(X_tech, dtype=torch.float32)).squeeze().numpy()
            preds_plain = scaler_target.inverse_transform(preds_plain_scaled.reshape(-1, 1)).flatten()
            
            # 3. LSTM + Sentiment
            sent_lstm = LSTMSentiment(input_size=len(all_features))
            sent_lstm.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker}_sent_lstm_h{h}.pt")))
            sent_lstm.eval()
            with torch.no_grad():
                preds_sent_scaled = sent_lstm(torch.tensor(X_all, dtype=torch.float32)).squeeze().numpy()
            preds_sent = scaler_target.inverse_transform(preds_sent_scaled.reshape(-1, 1)).flatten()
            
            # 4. Regime MoE
            moe_model = RegimeConditionedMoE(input_size=len(moe_features), sentiment_feature_idx=sentiment_idx)
            moe_model.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker}_moe_v2_h{h}.pt")))
            moe_model.eval()
            with torch.no_grad():
                preds_moe_scaled = moe_model(
                    torch.tensor(X_moe, dtype=torch.float32),
                    torch.tensor(r_probs, dtype=torch.float32)
                ).squeeze().numpy()
            preds_moe = scaler_target.inverse_transform(preds_moe_scaled.reshape(-1, 1)).flatten()
            
            # Calculate metrics per stock
            metrics_moe = calculate_metrics(actual_returns, preds_moe)
            metrics_plain = calculate_metrics(actual_returns, preds_plain)
            metrics_lr = calculate_metrics(actual_returns, preds_lr)
            metrics_sent = calculate_metrics(actual_returns, preds_sent)
            
            per_stock_scorecard.append({"Stock": ticker, "Horizon": f"{h}d", "Model": "Linear_Regression", **metrics_lr})
            per_stock_scorecard.append({"Stock": ticker, "Horizon": f"{h}d", "Model": "Plain_LSTM", **metrics_plain})
            per_stock_scorecard.append({"Stock": ticker, "Horizon": f"{h}d", "Model": "LSTM_Sentiment", **metrics_sent})
            per_stock_scorecard.append({"Stock": ticker, "Horizon": f"{h}d", "Model": "Regime_MoE", **metrics_moe})
            
            # Accumulate overall targets and predictions
            if ticker not in h_actuals:
                h_actuals[ticker] = actual_returns
                h_overall_preds[ticker] = {
                    "Linear_Regression": preds_lr,
                    "Plain_LSTM": preds_plain,
                    "LSTM_Sentiment": preds_sent,
                    "Regime_MoE": preds_moe
                }
                
            # Slice by regimes
            for idx, state in enumerate(regimes):
                state = int(state)
                per_regime_true[state].append(actual_returns[idx])
                per_regime_preds[state]["Linear_Regression"].append(preds_lr[idx])
                per_regime_preds[state]["Plain_LSTM"].append(preds_plain[idx])
                per_regime_preds[state]["LSTM_Sentiment"].append(preds_sent[idx])
                per_regime_preds[state]["Regime_MoE"].append(preds_moe[idx])
                
            # Slice transition-periods (within +/- 5 days of HMM switches)
            transition_indices = []
            for i in range(1, len(regimes)):
                if regimes[i] != regimes[i - 1]:
                    for offset in range(-5, 6):
                        offset_idx = i + offset
                        if 0 <= offset_idx < len(regimes):
                            transition_indices.append(offset_idx)
            transition_indices = sorted(list(set(transition_indices)))
            
            for idx in transition_indices:
                transition_true_list.append(actual_returns[idx])
                transition_preds_list["Linear_Regression"].append(preds_lr[idx])
                transition_preds_list["Plain_LSTM"].append(preds_plain[idx])
                transition_preds_list["LSTM_Sentiment"].append(preds_sent[idx])
                transition_preds_list["Regime_MoE"].append(preds_moe[idx])
                
            # -------------------------------------------------------------
            # Run Backtests
            # -------------------------------------------------------------
            # 1. Backtest with confidence gate
            bt_gated = run_backtest(test_df_h, preds_moe, h, ticker, use_confidence_gate=True)
            # 2. Backtest without confidence gate (Ablation study)
            bt_ungated = run_backtest(test_df_h, preds_moe, h, ticker, use_confidence_gate=False)
            
            backtest_scorecard.append({
                "Stock": ticker, "Horizon": f"{h}d", "Configuration": "Gated (Proposed)",
                "Sharpe": bt_gated["sharpe"], "Max Drawdown": bt_gated["max_dd"], "Hit Rate (%)": bt_gated["hit_rate"], "Signals": bt_gated["total_signals"]
            })
            backtest_scorecard.append({
                "Stock": ticker, "Horizon": f"{h}d", "Configuration": "Ungated (Ablation)",
                "Sharpe": bt_ungated["sharpe"], "Max Drawdown": bt_ungated["max_dd"], "Hit Rate (%)": bt_ungated["hit_rate"], "Signals": bt_ungated["total_signals"]
            })
            
            # Plot backtest curve for Gated MoE
            fig, ax = plt.subplots(figsize=(8, 4))
            dates_gated = test_df_h["date"].values[29 : 29 + len(bt_gated["equity_curve"])]
            dates_ungated = test_df_h["date"].values[29 : 29 + len(bt_ungated["equity_curve"])]
            ax.plot(dates_gated, bt_gated["equity_curve"], color="#17becf", label=f"MoE Gated (Sharpe: {bt_gated['sharpe']:.2f})")
            ax.plot(dates_ungated, bt_ungated["equity_curve"], color="#7f7f7f", linestyle="--", alpha=0.7, label=f"MoE Ungated (Sharpe: {bt_ungated['sharpe']:.2f})")
            ax.set_title(f"{ticker} Simulated swing Trading Equity Curves ({h}-Day Horizon)\n(Initial Cash: ₹100,000 | 15 bps Costs | 1-Day Lag)", color="white", fontsize=11)
            ax.set_ylabel("Portfolio Value (INR)", color="white")
            ax.legend(facecolor="#151722", labelcolor="white")
            fig.savefig(os.path.join(fig_dir, f"{ticker}_backtest_equity_h{h}.png"), dpi=150)
            plt.close(fig)

        # Layer 1 Metrics: Compile overall metrics aggregated across stocks
        all_true = np.concatenate(list(h_actuals.values()))
        for model in ["Linear_Regression", "Plain_LSTM", "LSTM_Sentiment", "Regime_MoE"]:
            all_preds = np.concatenate([h_overall_preds[t][model] for t in h_overall_preds])
            metrics = calculate_metrics(all_true, all_preds)
            overall_scorecard.append({"Horizon": f"{h}d", "Model": model, **metrics})
            
        # Layer 2 Metrics: Compile per-regime metrics
        regime_names = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
        for state, name in regime_names.items():
            true_state = np.array(per_regime_true[state])
            if len(true_state) > 0:
                for model in ["Linear_Regression", "Plain_LSTM", "LSTM_Sentiment", "Regime_MoE"]:
                    preds_state = np.array(per_regime_preds[state][model])
                    metrics = calculate_metrics(true_state, preds_state)
                    regime_scorecard.append({"Horizon": f"{h}d", "Regime": name, "Model": model, **metrics})
                    
        # Layer 3 Metrics: Transition-Period Headline Metrics
        true_trans = np.array(transition_true_list)
        if len(true_trans) > 0:
            for model in ["Linear_Regression", "Plain_LSTM", "LSTM_Sentiment", "Regime_MoE"]:
                preds_trans = np.array(transition_preds_list[model])
                metrics = calculate_metrics(true_trans, preds_trans)
                transition_scorecard.append({"Horizon": f"{h}d", "Model": model, **metrics})
                
    # Save Tables to CSV & Markdown
    # 1. Overall Scorecard
    df_overall = pd.DataFrame(overall_scorecard)
    df_overall.to_csv(os.path.join(tab_dir, "overall_results_v2.csv"), index=False)
    with open(os.path.join(tab_dir, "overall_results_v2.md"), "w", encoding="utf-8") as f:
        f.write("# Layer 1: Overall Accuracy Scorecard (Residual Returns)\n\n")
        f.write(df_overall.to_markdown(index=False))
        
    # 2. Per-Regime Scorecard
    df_regimes = pd.DataFrame(regime_scorecard)
    df_regimes.to_csv(os.path.join(tab_dir, "per_regime_results_v2.csv"), index=False)
    with open(os.path.join(tab_dir, "per_regime_results_v2.md"), "w", encoding="utf-8") as f:
        f.write("# Layer 2: Per-Regime Accuracy Scorecard\n\n")
        f.write(df_regimes.to_markdown(index=False))
        
    # 3. Transition-Period Headline Table
    df_transition = pd.DataFrame(transition_scorecard)
    df_transition.to_csv(os.path.join(tab_dir, "transition_period_results_v2.csv"), index=False)
    with open(os.path.join(tab_dir, "transition_period_results_v2.md"), "w", encoding="utf-8") as f:
        f.write("# Layer 3: Transition-Period Accuracy (+/- 5 Days) (Headline Metric)\n\n")
        f.write(df_transition.to_markdown(index=False))
        
    # 4. Per-Stock breakdown
    df_per_stock = pd.DataFrame(per_stock_scorecard)
    df_per_stock.to_csv(os.path.join(tab_dir, "per_stock_breakdown_v2.csv"), index=False)
    with open(os.path.join(tab_dir, "per_stock_breakdown_v2.md"), "w", encoding="utf-8") as f:
        f.write("# Layer 4: Per-Stock Performance breakdown\n\n")
        f.write(df_per_stock.to_markdown(index=False))
        
    # 5. Backtest results
    df_backtest = pd.DataFrame(backtest_scorecard)
    df_backtest.to_csv(os.path.join(tab_dir, "backtest_results_v2.csv"), index=False)
    with open(os.path.join(tab_dir, "backtest_results_v2.md"), "w", encoding="utf-8") as f:
        f.write("# Layer 5: Buy/Sell/Hold Backtest Performance & Ablation Study\n\n")
        f.write(df_backtest.to_markdown(index=False))
        
    # -------------------------------------------------------------
    # Grouped Bar Charts for Layer 1 Accuracy Comparison
    # -------------------------------------------------------------
    for h in horizons:
        df_sub = df_overall[df_overall["Horizon"] == f"{h}d"]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        sns.barplot(data=df_sub, x="Model", y="RMSE", palette="viridis", ax=ax)
        ax.set_ylabel("Root Mean Squared Error (RMSE)", color="white")
        ax.set_xlabel("Forecasting Model", color="white")
        ax.set_title(f"Forecasting Model Accuracy Comparison (RMSE, {h}-Day Horizon)", color="white", fontsize=12)
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"overall_accuracy_v2_h{h}.png"), dpi=150)
        plt.close(fig)
        
    # Print transition results
    print("\n" + "="*50 + "\nEVALUATION LAYER 3: TRANSITION-PERIOD ACCURACY (Weighted MoE v2) (±5 Days)\n" + "="*50)
    for index, row in df_transition.iterrows():
        print(f"Horizon: {row['Horizon']:<4} | Model: {row['Model']:<20} -> RMSE: {row['RMSE']:.4f} | MAE: {row['MAE']:.4f} | MAPE: {row['MAPE']:.2f}%")
        
    print("\nEvaluation complete! Tables and figures saved to outputs/")

if __name__ == "__main__":
    main()
