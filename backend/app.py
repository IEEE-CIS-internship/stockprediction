import os
import sys
import joblib
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any

# Ensure we can import model architectures
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "src", "models"))
from lstm import PlainLSTM
from lstm_sentiment import LSTMSentiment
from regime_gated import RegimeConditionedMoE

app = FastAPI(
    title="AuraTrade AI Backend Service",
    description="FastAPI REST service providing real-time swing trading forecasts, HMM regimes, and explainability",
    version="1.0.0"
)

# Enable CORS for frontend dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "data/processed/features/"
MODELS_DIR = "models/"
TABLES_DIR = "outputs/tables/"

# Load lists
def get_available_tickers():
    if not os.path.exists(DATA_DIR):
        return []
    return [f.split("_features.csv")[0] for f in os.listdir(DATA_DIR) if f.endswith("_features.csv") and not f.startswith("index")]

def load_stock_features(ticker: str):
    file_path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in database.")
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Live Inference Helpers
def run_moe_inference(ticker: str, horizon: int, df_stock: pd.DataFrame):
    if len(df_stock) < 30:
        raise HTTPException(status_code=400, detail="Insufficient stock historical data (minimum 30 days required).")
        
    last_seq_df = df_stock.iloc[-30:].copy()
    
    # Feature list
    moe_features = [
        "open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30",
        "sentiment_score", "vix", "crude", "usdinr", "is_split_day", "is_dividend_day", "days_to_earnings"
    ]
    sentiment_idx = moe_features.index("sentiment_score")
    
    # Load scaler & model weights
    scaler_moe_file = os.path.join(MODELS_DIR, f"{ticker}_scaler_moe.pkl")
    scaler_target_file = os.path.join(MODELS_DIR, f"{ticker}_scaler_target_h{horizon}.pkl")
    moe_model_file = os.path.join(MODELS_DIR, f"{ticker}_moe_h{horizon}.pt")
    
    if not os.path.exists(scaler_moe_file) or not os.path.exists(moe_model_file):
        raise HTTPException(status_code=500, detail=f"Model checkpoints for {ticker} h={horizon}d are missing on server.")
        
    scaler_moe = joblib.load(scaler_moe_file)
    scaler_target = joblib.load(scaler_target_file)
    
    # Scale features
    scaled_seq = last_seq_df.copy()
    scaled_seq[moe_features] = scaler_moe.transform(scaled_seq[moe_features])
    
    # Sequence tensor
    x = torch.tensor(scaled_seq[moe_features].values, dtype=torch.float32).unsqueeze(0)
    x.requires_grad = True
    
    # Regime probabilities
    r_probs = torch.tensor(last_seq_df[["prob_bull", "prob_bear", "prob_side"]].values[-1], dtype=torch.float32).unsqueeze(0)
    
    # Load Model
    model = RegimeConditionedMoE(input_size=len(moe_features), sentiment_feature_idx=sentiment_idx)
    model.load_state_dict(torch.load(moe_model_file))
    model.eval()
    
    # Forward pass and gradient capture
    pred = model(x, r_probs)
    pred.backward()
    
    # Attributions
    attributions = (x.grad * x).squeeze().detach().numpy()
    avg_attributions = np.mean(attributions, axis=0)
    total_attr = np.sum(np.abs(avg_attributions))
    attr_percentages = (avg_attributions / total_attr) * 100 if total_attr > 0 else avg_attributions
    
    feature_attributions = {feat: float(score) for feat, score in zip(moe_features, attr_percentages)}
    sorted_features = sorted(feature_attributions.items(), key=lambda item: abs(item[1]), reverse=True)
    
    pred_res_return = float(scaler_target.inverse_transform(pred.detach().numpy().reshape(-1, 1))[0, 0])
    
    return pred_res_return, sorted_features, last_seq_df, feature_attributions

# Schemas
class TickerResponse(BaseModel):
    tickers: List[str]

@app.get("/", tags=["General"])
def read_root():
    return {"status": "running", "service": "AuraTrade AI REST API"}

@app.get("/api/tickers", response_model=TickerResponse, tags=["Metadata"])
def get_tickers():
    return {"tickers": get_available_tickers()}

@app.get("/api/data/{ticker}", tags=["Market Data"])
def get_stock_data(ticker: str, limit: int = 100):
    df = load_stock_features(ticker)
    # Take the last 'limit' rows
    recent_df = df.tail(limit).copy()
    
    # Format dates as strings
    recent_df["date"] = recent_df["date"].dt.strftime("%Y-%m-%d")
    
    # Convert to list of dicts
    data = recent_df[["date", "open", "high", "low", "close", "volume", "returns", "sentiment_score", "is_split_day", "is_dividend_day", "days_to_earnings"]].to_dict(orient="records")
    return {"ticker": ticker, "records_count": len(data), "data": data}

@app.get("/api/regime/{ticker}", tags=["Market Regime"])
def get_regime_info(ticker: str):
    df = load_stock_features(ticker)
    
    # Active HMM details
    last_row = df.iloc[-1]
    active_regime_idx = int(last_row["regime"])
    regimes_mapping = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
    
    # Transition duration statistics from outputs/tables if present
    duration_file = os.path.join("outputs/tables/", "regime_duration_stats.csv")
    duration_stats = {}
    if os.path.exists(duration_file):
        df_dur = pd.read_csv(duration_file)
        # Find stats for this ticker
        ticker_stats = df_dur[df_dur["Stock"] == ticker]
        if not ticker_stats.empty:
            duration_stats = ticker_stats.to_dict(orient="records")[0]
            
    return {
        "ticker": ticker,
        "date": last_row["date"].strftime("%Y-%m-%d"),
        "active_regime": regimes_mapping[active_regime_idx],
        "regime_index": active_regime_idx,
        "probabilities": {
            "Bullish": float(last_row["prob_bull"]),
            "Bearish": float(last_row["prob_bear"]),
            "Sideways": float(last_row["prob_side"])
        },
        "duration_statistics": duration_stats
    }
@app.post("/api/refresh/{ticker}", tags=["Market Data"])
def refresh_ticker_data(ticker: str):
    import subprocess
    if ticker not in get_available_tickers():
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} is not in the system's stock list.")
        
    try:
        # 1. Run Data Pipeline to download and cache latest daily data
        subprocess.run([
            sys.executable, "src/data_pipeline.py",
            "--tickers", ticker,
            "--index", "^NSEI",
            "--years", "5",
            "--out", "data/raw/"
        ], check=True, capture_output=True, text=True)
        
        # 2. Run Regime labeling HMM
        subprocess.run([
            sys.executable, "src/regime_hmm.py",
            "--in", "data/raw/",
            "--out", "data/processed/regimes/"
        ], check=True, capture_output=True, text=True)
        
        # 3. Run Feature Engineering to update processed features
        subprocess.run([
            sys.executable, "src/build_features.py",
            "--in", "data/raw/", "data/processed/regimes/",
            "--out", "data/processed/features/"
        ], check=True, capture_output=True, text=True)
        
        # Reload features to verify updated row count and get new details
        df = load_stock_features(ticker)
        last_row = df.iloc[-1]
        active_regime_idx = int(last_row["regime"])
        regimes_mapping = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
        active_regime = regimes_mapping[active_regime_idx]
        
        return {
            "status": "success",
            "message": f"Successfully pulled latest market data and recomputed HMM regimes/features for {ticker}.",
            "ticker": ticker,
            "updated_records_count": len(df),
            "last_date": last_row["date"].strftime("%Y-%m-%d"),
            "close_price": float(last_row["close"]),
            "current_regime": active_regime
        }
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Data refresh failed during processing pipeline: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")


@app.get("/api/forecast/{ticker}", tags=["Forecasting"])
def get_forecast(ticker: str, horizon: int = Query(3, ge=1, le=7)):
    if horizon not in [1, 3, 7]:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 3, or 7 days.")
        
    df = load_stock_features(ticker)
    pred_return, _, last_seq_df, _ = run_moe_inference(ticker, horizon, df)
    
    last_row = last_seq_df.iloc[-1]
    active_regime_idx = int(last_row["regime"])
    regimes_mapping = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
    active_regime = regimes_mapping[active_regime_idx]
    confidence = float(last_row[f"prob_{['bull', 'bear', 'side'][active_regime_idx]}"])
    
    # Calculate trading signals based on return volatility
    ret_std = df["returns"].std()
    threshold_buy = ret_std * 0.5
    threshold_sell = -ret_std * 0.5
    
    signal = "HOLD"
    if pred_return > threshold_buy and active_regime_idx != 1 and confidence >= 0.50:
        signal = "BUY"
    elif pred_return < threshold_sell or (active_regime_idx == 1 and confidence >= 0.50):
        signal = "SELL"
        
    return {
        "ticker": ticker,
        "horizon": f"{horizon}d",
        "predicted_residual_return": pred_return,
        "volatility_threshold_buy": threshold_buy,
        "volatility_threshold_sell": threshold_sell,
        "active_regime": active_regime,
        "regime_confidence": confidence,
        "trading_signal": signal,
        "as_of_date": last_row["date"].strftime("%Y-%m-%d")
    }

@app.get("/api/explain/{ticker}", tags=["Explainability"])
def get_explanation(ticker: str, horizon: int = Query(3, ge=1, le=7)):
    if horizon not in [1, 3, 7]:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 3, or 7 days.")
        
    df = load_stock_features(ticker)
    pred_return, sorted_features, last_seq_df, feature_attributions = run_moe_inference(ticker, horizon, df)
    
    last_row = last_seq_df.iloc[-1]
    active_regime_idx = int(last_row["regime"])
    regimes_mapping = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
    active_regime = regimes_mapping[active_regime_idx]
    confidence = float(last_row[f"prob_{['bull', 'bear', 'side'][active_regime_idx]}"])
    
    ret_std = df["returns"].std()
    threshold_buy = ret_std * 0.5
    threshold_sell = -ret_std * 0.5
    
    signal = "HOLD"
    if pred_return > threshold_buy and active_regime_idx != 1 and confidence >= 0.50:
        signal = "BUY"
    elif pred_return < threshold_sell or (active_regime_idx == 1 and confidence >= 0.50):
        signal = "SELL"
        
    # Generate Reasoning
    from explain import generate_recommendation_text
    reasoning_text = generate_recommendation_text(ticker, horizon, signal, active_regime, confidence, sorted_features)
    
    return {
        "ticker": ticker,
        "horizon": f"{horizon}d",
        "predicted_residual_return": pred_return,
        "active_regime": active_regime,
        "regime_confidence": confidence,
        "trading_signal": signal,
        "feature_attributions": feature_attributions,
        "top_drivers": [
            {"feature": f, "importance_score": float(val)} for f, val in sorted_features[:5]
        ],
        "recommendation_reasoning": reasoning_text,
        "as_of_date": last_row["date"].strftime("%Y-%m-%d")
    }

@app.get("/api/backtest/{ticker}", tags=["Performance"])
def get_backtest_metrics(ticker: str, horizon: int = Query(3, ge=1, le=7)):
    backtest_file = os.path.join(TABLES_DIR, "backtest_results.csv")
    if not os.path.exists(backtest_file):
        raise HTTPException(status_code=404, detail="Backtest results are not compiled yet on the server.")
        
    df_bt = pd.read_csv(backtest_file)
    ticker_bt = df_bt[(df_bt["Stock"] == ticker) & (df_bt["Horizon"] == f"{horizon}d")]
    
    if ticker_bt.empty:
        raise HTTPException(status_code=404, detail=f"No backtest metrics found for {ticker} h={horizon}d.")
        
    records = ticker_bt.to_dict(orient="records")
    gated_record = next((r for r in records if "Gated" in r["Configuration"]), None)
    ungated_record = next((r for r in records if "Ungated" in r["Configuration"]), None)
    
    return {
        "ticker": ticker,
        "horizon": f"{horizon}d",
        "gated_configuration": gated_record,
        "ungated_configuration": ungated_record
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
