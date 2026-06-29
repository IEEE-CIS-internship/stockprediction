import os
import argparse
import json
import datetime
import joblib
import pandas as pd
import numpy as np
import torch

# Import MoE architecture
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from regime_gated import RegimeConditionedMoE

def generate_recommendation_text(ticker, horizon, signal, regime, confidence, top_features):
    """
    Generates a structured trading recommendation reasoning paragraph.
    """
    regime_str = regime.capitalize()
    conf_pct = confidence * 100
    
    feature_drivers = ", ".join([f"{feat} ({weight:+.2f}%)" for feat, weight in top_features[:3]])
    
    if signal == "BUY":
        text = (
            f"RECOMMENDATION: BUY {ticker} for a {horizon}-day swing horizon. "
            f"The underlying HMM classifies the market state as {regime_str} with {conf_pct:.1f}% confidence. "
            f"Our Regime-Conditioned Mixture-of-Experts forecasts a positive residual return relative to the NIFTY 50 index. "
            f"The primary quantitative drivers of this bullish signal are: {feature_drivers}. "
            f"Risk note: Maintain trailing stop-loss as transaction fees (15 bps) and execution lags are factored in."
        )
    elif signal == "SELL":
        text = (
            f"RECOMMENDATION: SELL/SHORT {ticker} for a {horizon}-day swing horizon. "
            f"The underlying HMM classifies the market state as {regime_str} with {conf_pct:.1f}% confidence. "
            f"The Regime-Conditioned MoE model predicts a negative residual return (underperformance relative to index). "
            f"The primary quantitative drivers of this bearish signal are: {feature_drivers}. "
            f"Action: Reduce long exposure or establish short positions where appropriate."
        )
    else:
        text = (
            f"RECOMMENDATION: HOLD/NEUTRAL for {ticker}. "
            f"The HMM state is {regime_str} ({conf_pct:.1f}% confidence). "
            f"The predicted residual return does not cross the volatility-defined entry thresholds. "
            f"Primary neutral drivers: {feature_drivers}. "
            f"Action: Keep capital in cash and monitor for a clearer signal."
        )
    return text

def main():
    parser = argparse.ArgumentParser(description="AuraTrade AI Explainability Layer")
    parser.add_argument("--ticker", default="ICICIBANK.NS", help="Ticker to explain")
    parser.add_argument("--horizon", type=int, default=3, help="Forecast horizon (1, 3, or 7)")
    parser.add_argument("--out", default="outputs/explanations/", help="Output directory")
    
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    features_dir = "data/processed/features/"
    models_dir = "models"
    
    ticker = args.ticker
    h = args.horizon
    
    features_file = os.path.join(features_dir, f"{ticker}_features.csv")
    if not os.path.exists(features_file):
        print(f"Error: Feature file {features_file} not found.")
        return
        
    df = pd.read_csv(features_file)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Slice the last sequence (last 30 trading days)
    if len(df) < 30:
        print("Error: Dataset has less than 30 rows.")
        return
        
    last_seq_df = df.iloc[-30:].copy()
    
    # Feature list for MoE
    moe_features = [
        "open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30",
        "sentiment_score", "vix", "crude", "usdinr", "is_split_day", "is_dividend_day", "days_to_earnings"
    ]
    sentiment_idx = moe_features.index("sentiment_score")
    
    # Load input and target scalers
    scaler_moe_file = os.path.join(models_dir, f"{ticker}_scaler_moe.pkl")
    scaler_target_file = os.path.join(models_dir, f"{ticker}_scaler_target_h{h}.pkl")
    moe_model_file = os.path.join(models_dir, f"{ticker}_moe_h{h}.pt")
    
    if not os.path.exists(scaler_moe_file) or not os.path.exists(moe_model_file):
        print("Error: Model checkpoints not found. Run training scripts first.")
        return
        
    scaler_moe = joblib.load(scaler_moe_file)
    scaler_target = joblib.load(scaler_target_file)
    
    # Scale features
    scaled_seq = last_seq_df.copy()
    scaled_seq[moe_features] = scaler_moe.transform(scaled_seq[moe_features])
    
    # Convert to sequence tensor: shape (1, 30, 16)
    x = torch.tensor(scaled_seq[moe_features].values, dtype=torch.float32).unsqueeze(0)
    x.requires_grad = True
    
    # Regime probabilities for gating at the last day (t + lookback - 1)
    # shape: (1, 3)
    r_probs = torch.tensor(last_seq_df[["prob_bull", "prob_bear", "prob_side"]].values[-1], dtype=torch.float32).unsqueeze(0)
    
    # Load Model
    model = RegimeConditionedMoE(input_size=len(moe_features), sentiment_feature_idx=sentiment_idx)
    model.load_state_dict(torch.load(moe_model_file))
    model.eval()
    
    # Forward pass and gradient capture
    pred = model(x, r_probs)
    pred.backward()
    
    # Compute Vanilla Gradient * Input attribution: shape (1, 30, 16)
    attributions = (x.grad * x).squeeze().detach().numpy() # shape (30, 16)
    
    # Average attribution of each feature over the sequence
    avg_attributions = np.mean(attributions, axis=0)
    
    # Normalize attributions to percentages for readability
    total_attr = np.sum(np.abs(avg_attributions))
    if total_attr > 0:
        attr_percentages = (avg_attributions / total_attr) * 100
    else:
        attr_percentages = avg_attributions
        
    feature_attributions = {}
    for name, score in zip(moe_features, attr_percentages):
        feature_attributions[name] = float(score)
        
    # Sort features by absolute attribution
    sorted_features = sorted(feature_attributions.items(), key=lambda item: abs(item[1]), reverse=True)
    
    # Determine forecast signal
    pred_res_return = float(scaler_target.inverse_transform(pred.detach().numpy().reshape(-1, 1))[0, 0])
    ret_std = df["returns"].std()
    
    regimes_mapping = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
    active_regime_idx = int(last_seq_df["regime"].values[-1])
    active_regime = regimes_mapping[active_regime_idx]
    
    probs = last_seq_df[["prob_bull", "prob_bear", "prob_side"]].values[-1]
    confidence = float(probs[active_regime_idx])
    
    # Signals
    if pred_res_return > ret_std * 0.5 and active_regime_idx != 1 and confidence >= 0.50:
        signal = "BUY"
    elif pred_res_return < -ret_std * 0.5 or (active_regime_idx == 1 and confidence >= 0.50):
        signal = "SELL"
    else:
        signal = "HOLD"
        
    recommendation_text = generate_recommendation_text(ticker, h, signal, active_regime, confidence, sorted_features)
    
    # Assemble JSON object
    explainability_payload = {
        "ticker": ticker,
        "date": last_seq_df["date"].iloc[-1].strftime("%Y-%m-%d"),
        "horizon": f"{h}d",
        "predicted_residual_return": pred_res_return,
        "active_regime": active_regime,
        "regime_confidence": confidence,
        "trading_signal": signal,
        "feature_attributions": feature_attributions,
        "top_drivers": [
            {"feature": f, "importance_score": float(val)} for f, val in sorted_features[:5]
        ],
        "gating_weights": {
            "Bullish": float(probs[0]),
            "Bearish": float(probs[1]),
            "Sideways": float(probs[2])
        },
        "recommendation_reasoning": recommendation_text
    }
    
    # Save ticker specific explanation
    out_file = os.path.join(args.out, f"{ticker}_explainability.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(explainability_payload, f, indent=4)
        
    # Save global sample_explainability.json (Step 9 specific deliverable)
    sample_file = os.path.join(args.out, "sample_explainability.json")
    with open(sample_file, "w", encoding="utf-8") as f:
        json.dump(explainability_payload, f, indent=4)
        
    print(f"\n--- Explainability Suite for {ticker} (Horizon: {h}d) ---")
    print(f"Signal Generated: {signal}")
    print(f"Regime: {active_regime} (Confidence: {confidence*100:.1f}%)")
    print(f"Reasoning:\n{recommendation_text}")
    print("\nFeature Attributions (Top 5 Drivers):")
    for feat, score in sorted_features[:5]:
        print(f"  - {feat:<20}: {score:+.2f}%")
        
    print("\nExplainability payload saved to:", sample_file)

if __name__ == "__main__":
    main()
