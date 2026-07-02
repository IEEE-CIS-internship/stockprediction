import os
import sys
import joblib
import datetime as dt
import subprocess
import threading
import time
import pandas as pd
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
from zoneinfo import ZoneInfo

# Ensure we can import model architectures
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)
from models import PlainLSTM, LSTMSentiment, RegimeConditionedMoE

STOCK_UNIVERSE = [
    "ICICIBANK.NS",
    "HCLTECH.NS",
    "INDIGO.NS",
    "MARUTI.NS",
    "TRENT.NS",
    "HINDALCO.NS",
    "ONGC.NS",
    "ADANIENT.NS",
]

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

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
REGIMES_DIR = os.path.join(BASE_DIR, "data", "processed", "regimes")
DATA_DIR = os.path.join(BASE_DIR, "data", "processed", "features")
MODELS_DIR = os.path.join(BASE_DIR, "models")
TABLES_DIR = os.path.join(BASE_DIR, "outputs", "tables")

_refresh_lock = threading.Lock()

# Load lists
def get_available_tickers():
    if not os.path.exists(DATA_DIR):
        return []
    return [f.split("_features.csv")[0] for f in os.listdir(DATA_DIR) if f.endswith("_features.csv") and not f.startswith("index")]

def load_stock_features(ticker: str):
    file_path = os.path.join(DATA_DIR, f"{ticker}_features.csv")
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=404,
            detail=(
                f"Ticker {ticker} not found in local feature database. "
                "Run POST /api/refresh/all or POST /api/refresh/{ticker} to fetch yFinance data first."
            ),
        )
    df = pd.read_csv(file_path)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# Live Inference Helpers
def run_statistical_inference(ticker: str, horizon: int, df_stock: pd.DataFrame):
    if len(df_stock) < 30:
        raise HTTPException(status_code=400, detail="Insufficient stock historical data (minimum 30 days required).")

    target_col = f"target_res_return_{horizon}d"
    last_seq_df = df_stock.iloc[-30:].copy()
    target_history = df_stock[target_col].dropna() if target_col in df_stock.columns else pd.Series(dtype=float)

    if target_history.empty:
        recent_residual = df_stock["returns"].tail(20).mean() if "returns" in df_stock.columns else 0.0
    else:
        recent_residual = target_history.tail(60).ewm(span=15, adjust=False).mean().iloc[-1]

    last_row = last_seq_df.iloc[-1]
    regime_idx = int(last_row["regime"])
    regime_bias = {0: 0.0015, 1: -0.0015, 2: 0.0}.get(regime_idx, 0.0)
    pred_res_return = float(recent_residual + regime_bias)

    feature_attributions = {
        "returns": 35.0,
        "volatility_30": 20.0,
        "sentiment_score": 10.0,
        "prob_bull": float(last_row.get("prob_bull", 0.0) * 20.0),
        "prob_bear": float(-last_row.get("prob_bear", 0.0) * 20.0),
        "prob_side": float(last_row.get("prob_side", 0.0) * 10.0),
    }
    sorted_features = sorted(feature_attributions.items(), key=lambda item: abs(item[1]), reverse=True)
    return pred_res_return, sorted_features, last_seq_df, feature_attributions, "statistical_fallback"

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
    
    if not os.path.exists(scaler_moe_file) or not os.path.exists(scaler_target_file) or not os.path.exists(moe_model_file):
        return run_statistical_inference(ticker, horizon, df_stock)
        
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
    model = RegimeConditionedMoE(input_size=len(moe_features))
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
    
    return pred_res_return, sorted_features, last_seq_df, feature_attributions, "regime_moe"

# Schemas
class TickerResponse(BaseModel):
    tickers: List[str]

def refresh_market_data(tickers: List[str]) -> Dict[str, Any]:
    invalid = sorted(set(tickers) - set(STOCK_UNIVERSE))
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unsupported ticker(s): {', '.join(invalid)}")

    with _refresh_lock:
        for folder in [RAW_DIR, REGIMES_DIR, DATA_DIR, TABLES_DIR, os.path.join(BASE_DIR, "outputs", "figures")]:
            os.makedirs(folder, exist_ok=True)

        commands = [
            [
                sys.executable,
                os.path.join(SRC_DIR, "data_pipeline.py"),
                "--tickers",
                *tickers,
                "--index",
                "^NSEI",
                "--years",
                "5",
                "--out",
                RAW_DIR,
            ],
            [
                sys.executable,
                os.path.join(SRC_DIR, "regime_hmm.py"),
                "--in",
                RAW_DIR,
                "--out",
                REGIMES_DIR,
            ],
            [
                sys.executable,
                os.path.join(SRC_DIR, "build_features.py"),
                "--in",
                RAW_DIR,
                REGIMES_DIR,
                "--out",
                DATA_DIR,
            ],
        ]

        logs = []
        for command in commands:
            completed = subprocess.run(command, cwd=BASE_DIR, check=True, capture_output=True, text=True)
            logs.append(completed.stdout)

    refreshed = get_available_tickers()
    return {
        "status": "success",
        "requested_tickers": tickers,
        "available_tickers": refreshed,
        "refreshed_at": dt.datetime.now(ZoneInfo("Asia/Kolkata")).isoformat(timespec="seconds"),
        "log_tail": "\n".join(logs)[-4000:],
    }

def latest_feature_date():
    dates = []
    if not os.path.exists(DATA_DIR):
        return None
    for filename in os.listdir(DATA_DIR):
        if not filename.endswith("_features.csv"):
            continue
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, filename), usecols=["date"])
            if not df.empty:
                dates.append(pd.to_datetime(df["date"]).max().date())
        except Exception:
            continue
    return max(dates) if dates else None

def is_market_day(day):
    return day.weekday() < 5

def scheduled_market_refresh():
    tz = ZoneInfo("Asia/Kolkata")
    while True:
        now = dt.datetime.now(tz)
        next_run = now.replace(hour=16, minute=0, second=0, microsecond=0)
        if now >= next_run:
            next_run += dt.timedelta(days=1)
        while not is_market_day(next_run.date()):
            next_run += dt.timedelta(days=1)
        time.sleep(max(60, (next_run - now).total_seconds()))

        latest = latest_feature_date()
        if latest is None or latest < next_run.date():
            try:
                refresh_market_data(STOCK_UNIVERSE)
            except Exception as exc:
                print(f"Scheduled yFinance refresh failed: {exc}", flush=True)

@app.on_event("startup")
def start_market_refresh_scheduler():
    thread = threading.Thread(target=scheduled_market_refresh, daemon=True)
    thread.start()

@app.get("/", tags=["General"])
def read_root():
    return {
        "status": "running",
        "service": "AuraTrade AI REST API",
        "available_tickers": get_available_tickers(),
        "latest_feature_date": str(latest_feature_date()) if latest_feature_date() else None,
    }

@app.get("/api/tickers", response_model=TickerResponse, tags=["Metadata"])
def get_tickers():
    tickers = get_available_tickers()
    return {"tickers": tickers if tickers else STOCK_UNIVERSE}

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
@app.post("/api/refresh/all", tags=["Market Data"])
def refresh_all_ticker_data():
    try:
        return refresh_market_data(STOCK_UNIVERSE)
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Data refresh failed during processing pipeline: {e.stderr}")

@app.post("/api/refresh/{ticker}", tags=["Market Data"])
def refresh_ticker_data(ticker: str):
    try:
        refresh_info = refresh_market_data([ticker])
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
            "current_regime": active_regime,
            "refresh": refresh_info,
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
    pred_return, _, last_seq_df, _, model_source = run_moe_inference(ticker, horizon, df)
    
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
        "model_source": model_source,
        "trading_signal": signal,
        "as_of_date": last_row["date"].strftime("%Y-%m-%d")
    }

@app.get("/api/explain/{ticker}", tags=["Explainability"])
def get_explanation(ticker: str, horizon: int = Query(3, ge=1, le=7)):
    if horizon not in [1, 3, 7]:
        raise HTTPException(status_code=400, detail="Horizon must be 1, 3, or 7 days.")
        
    df = load_stock_features(ticker)
    pred_return, sorted_features, last_seq_df, feature_attributions, model_source = run_moe_inference(ticker, horizon, df)
    
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
    if model_source == "statistical_fallback":
        reasoning_text = (
            "MODEL NOTICE: Trained MoE checkpoint files were not found, so this live signal uses a "
            "statistical residual-return fallback based on the refreshed yFinance features. "
            + reasoning_text.replace("Our Regime-Conditioned Mixture-of-Experts forecasts", "The fallback model estimates")
            .replace("The Regime-Conditioned MoE model predicts", "The fallback model estimates")
        )
    
    return {
        "ticker": ticker,
        "horizon": f"{horizon}d",
        "predicted_residual_return": pred_return,
        "active_regime": active_regime,
        "regime_confidence": confidence,
        "model_source": model_source,
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
