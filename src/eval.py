import os
import pandas as pd
import numpy as np
import torch
import joblib
from torch.utils.data import DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
from models import BaselineLinearRegression, PlainLSTM, LSTMSentiment, RegimeConditionedMoE
from train import create_sequences, StockSequenceDataset

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Filter out zero values to avoid division by zero
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def calculate_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {"RMSE": rmse, "MAE": mae, "MAPE": mape}

def run_evaluation(ticker_name, horizon=1, lookback=30):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_filepath = os.path.join(base_dir, "data", f"regime_{ticker_name}_processed.csv")
    models_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(data_filepath):
        print(f"Data file not found for {ticker_name}")
        return None
        
    df = pd.read_csv(data_filepath)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Handle missing news sentiment score if news fetching is sparse
    if "sentiment_score" not in df.columns:
        df["sentiment_score"] = np.random.uniform(-0.5, 0.5, len(df))
        
    # Technical Features list
    tech_features = ["open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30"]
    all_features = tech_features + ["sentiment_score"]
    
    # Chronological Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    test_df = df.iloc[split_idx:].copy()
    
    # Load Scalers
    try:
        scaler_tech = joblib.load(os.path.join(models_dir, f"{ticker_name}_scaler_tech.pkl"))
        scaler_all = joblib.load(os.path.join(models_dir, f"{ticker_name}_scaler_all.pkl"))
        target_scaler = joblib.load(os.path.join(models_dir, f"{ticker_name}_scaler_target.pkl"))
    except Exception as e:
        print(f"Error loading scalers for {ticker_name}: {e}. Ensure the model is trained first.")
        return None
        
    # Scale test set
    test_df[tech_features] = scaler_tech.transform(test_df[tech_features])
    test_df[all_features] = scaler_all.transform(test_df[all_features])
    test_df["close_scaled"] = target_scaler.transform(test_df[["close"]])
    
    # Prepare sequence arrays
    X_test_tech, y_test_scaled, _ = create_sequences(test_df, tech_features, target_col="close_scaled", lookback=lookback, horizon=horizon)
    X_test_all, _, r_test = create_sequences(test_df, all_features, target_col="close_scaled", lookback=lookback, horizon=horizon)
    
    # Check HMM regimes corresponding to the target dates
    # Target dates align with the end of the lookback sequence + horizon
    target_indices = np.arange(lookback - 1 + horizon, len(test_df))
    regimes = test_df.iloc[target_indices]["regime"].values
    actual_prices = target_scaler.inverse_transform(y_test_scaled.reshape(-1, 1)).flatten()
    
    # Load PyTorch Model Architectures
    device = torch.device("cpu")
    
    plain_lstm = PlainLSTM(input_size=len(tech_features))
    sent_lstm = LSTMSentiment(input_size=len(all_features))
    moe_model = RegimeConditionedMoE(input_size=len(all_features))
    
    try:
        plain_lstm.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker_name}_plain_lstm_h{horizon}.pt"), map_location=device))
        sent_lstm.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker_name}_sent_lstm_h{horizon}.pt"), map_location=device))
        moe_model.load_state_dict(torch.load(os.path.join(models_dir, f"{ticker_name}_moe_h{horizon}.pt"), map_location=device))
        
        plain_lstm.eval()
        sent_lstm.eval()
        moe_model.eval()
    except Exception as e:
        print(f"Error loading model weights for {ticker_name}: {e}. Ensure the models are fully trained.")
        return None
        
    # Generate Predictions
    preds = {}
    
    # 1. Baseline Linear Regression
    lr_model = BaselineLinearRegression()
    # Fit on training data is assumed completed in train.py, for evaluation we train a mock LR on sequence to verify
    # (Since train.py does not cache LR model on disk, we re-fit it quickly on the training split)
    train_df = df.iloc[:split_idx].copy()
    train_df[tech_features] = scaler_tech.transform(train_df[tech_features])
    train_df["close_scaled"] = target_scaler.transform(train_df[["close"]])
    X_train_tech, y_train_scaled, _ = create_sequences(train_df, tech_features, target_col="close_scaled", lookback=lookback, horizon=horizon)
    lr_model.fit(X_train_tech.reshape(len(X_train_tech), -1), y_train_scaled)
    
    preds_lr_scaled = lr_model.predict(X_test_tech.reshape(len(X_test_tech), -1))
    preds["Linear_Regression"] = target_scaler.inverse_transform(preds_lr_scaled.reshape(-1, 1)).flatten()
    
    # Convert sequence arrays to PyTorch tensors for evaluation
    t_X_tech = torch.tensor(X_test_tech, dtype=torch.float32)
    t_X_all = torch.tensor(X_test_all, dtype=torch.float32)
    t_r_test = torch.tensor(r_test, dtype=torch.float32)
    
    with torch.no_grad():
        # 2. Plain LSTM
        preds_plain_scaled = plain_lstm(t_X_tech).squeeze().numpy()
        preds["Plain_LSTM"] = target_scaler.inverse_transform(preds_plain_scaled.reshape(-1, 1)).flatten()
        
        # 3. LSTM + Sentiment
        preds_sent_scaled = sent_lstm(t_X_all).squeeze().numpy()
        preds["LSTM_Sentiment"] = target_scaler.inverse_transform(preds_sent_scaled.reshape(-1, 1)).flatten()
        
        # 4. Regime-Conditioned MoE
        preds_moe_scaled = moe_model(t_X_all, t_r_test).squeeze().numpy()
        preds["Regime_MoE"] = target_scaler.inverse_transform(preds_moe_scaled.reshape(-1, 1)).flatten()
        
    # Calculate Evaluation Layers
    
    # Layer 1: Overall Metrics
    overall_metrics = {}
    for model_name, y_pred in preds.items():
        overall_metrics[model_name] = calculate_metrics(actual_prices, y_pred)
        
    # Layer 2: Per-Regime Metrics
    # regimes contains values {0: Bull, 1: Bear, 2: Sideways}
    per_regime_metrics = {0: {}, 1: {}, 2: {}}
    for state in [0, 1, 2]:
        mask = regimes == state
        if np.sum(mask) > 0:
            for model_name, y_pred in preds.items():
                per_regime_metrics[state][model_name] = calculate_metrics(actual_prices[mask], y_pred[mask])
                
    # Layer 3: Transition-Period Metrics (Headline Differentiator)
    # Find transition indices in the full test set
    # A transition is where regime changes
    transition_indices = []
    # Check transitions inside the targeted evaluation subsegment
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i - 1]:
            # Add all indices within +/- 5 days of transition
            for offset in range(-5, 6):
                idx = i + offset
                if 0 <= idx < len(regimes):
                    transition_indices.append(idx)
                    
    transition_indices = sorted(list(set(transition_indices)))
    transition_metrics = {}
    
    if transition_indices:
        y_true_trans = actual_prices[transition_indices]
        for model_name, y_pred in preds.items():
            transition_metrics[model_name] = calculate_metrics(y_true_trans, y_pred[transition_indices])
            
    return {
        "overall": overall_metrics,
        "per_regime": per_regime_metrics,
        "transition": transition_metrics
    }

def print_evaluation_summary(eval_results):
    if eval_results is None:
        return
        
    print("\n" + "="*50 + "\nEVALUATION LAYER 1: OVERALL ACCURACY METRICS\n" + "="*50)
    for model, metrics in eval_results["overall"].items():
        print(f"{model:<20} -> RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | MAPE: {metrics['MAPE']:.2f}%")
        
    print("\n" + "="*50 + "\nEVALUATION LAYER 2: PER-REGIME ACCURACY\n" + "="*50)
    regime_names = {0: "BULLISH", 1: "BEARISH", 2: "SIDEWAYS"}
    for state, state_name in regime_names.items():
        print(f"\nRegime: {state_name}")
        for model, metrics in eval_results["per_regime"][state].items():
            print(f"  {model:<20} -> RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | MAPE: {metrics['MAPE']:.2f}%")
            
    print("\n" + "="*50 + "\nEVALUATION LAYER 3: TRANSITION-PERIOD ACCURACY (+/- 5 Days)\n" + "="*50)
    for model, metrics in eval_results["transition"].items():
        print(f"{model:<20} -> RMSE: {metrics['RMSE']:.4f} | MAE: {metrics['MAE']:.4f} | MAPE: {metrics['MAPE']:.2f}%")
        
if __name__ == "__main__":
    res = run_evaluation("ICICIBANK", horizon=1)
    if res:
        print_evaluation_summary(res)
