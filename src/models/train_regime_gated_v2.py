import os
import argparse
import datetime
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Import MoE architecture
from regime_gated import RegimeConditionedMoE

class StockSequenceMoeDatasetV2(Dataset):
    def __init__(self, X, regime_probs, y, weights):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.regime_probs = torch.tensor(regime_probs, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.regime_probs[idx], self.y[idx], self.weights[idx]

def create_sequences_moe_v2(df, features, target_col, lookback=30, horizon=1):
    X, regime_probs, y, sample_weights = [], [], [], []
    feature_data = df[features].values
    regime_data = df[["prob_bull", "prob_bear", "prob_side"]].values
    target_data = df[target_col].values
    transition_window = df["is_transition_window"].values
    
    for i in range(len(df) - lookback - horizon + 1):
        X.append(feature_data[i : i + lookback])
        regime_probs.append(regime_data[i + lookback - 1])
        y.append(target_data[i + lookback - 1 + horizon])
        # Weight is based on whether lookback end falls in transition window
        is_trans = transition_window[i + lookback - 1]
        sample_weights.append(2.0 if is_trans == 1.0 else 1.0)
        
    return np.array(X), np.array(regime_probs), np.array(y), np.array(sample_weights)

def train_moe_model_v2(model, dataloader, epochs=20, lr=0.001):
    # Reduction='none' so we can apply custom sample weights per row
    criterion = nn.MSELoss(reduction='none')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for batch_x, batch_regime, batch_y, batch_w in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x, batch_regime)
            raw_loss = criterion(predictions.squeeze(), batch_y)
            loss = torch.mean(raw_loss * batch_w)
            loss.backward()
            optimizer.step()
    return model

def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

def main():
    parser = argparse.ArgumentParser(description="Train Weighted Regime-Gated MoE Model")
    parser.add_argument("--in", dest="in_dir", required=True, help="Features directory")
    parser.add_argument("--out", nargs=2, required=True, help="Output directories: [exp_dir, tab_dir]")
    
    args = parser.parse_args()
    exp_dir, tab_dir = args.out[0], args.out[1]
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    moe_features = [
        "open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30",
        "sentiment_score", "vix", "crude", "usdinr", "is_split_day", "is_dividend_day", "days_to_earnings"
    ]
    sentiment_idx = moe_features.index("sentiment_score")
    
    horizons = [1, 3, 7]
    lookback = 30
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    
    log_file = os.path.join(exp_dir, "run_log_v2.csv")
    
    tickers = [f.split("_features.csv")[0] for f in os.listdir(args.in_dir) if f.endswith("_features.csv") and not f.startswith("index")]
    
    # Initialize run log v2 with header if not exists
    if not os.path.exists(log_file):
        pd.DataFrame(columns=["date", "ticker", "horizon", "model", "rmse", "mae", "mape"]).to_csv(log_file, index=False)
        
    for ticker in tickers:
        print(f"\n--- Training Weighted Regime-Gated MoE for {ticker} ---")
        
        # Load features
        features_file = os.path.join(args.in_dir, f"{ticker}_features.csv")
        df = pd.read_csv(features_file)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Identify transition periods in the original dataframe
        df["is_transition_window"] = 0.0
        transition_indices = df[df["regime"] != df["regime"].shift(1)].index
        transition_indices = [idx for idx in transition_indices if idx > 0]
        
        for idx in transition_indices:
            start_idx = max(0, idx - 5)
            end_idx = min(len(df) - 1, idx + 5)
            df.loc[start_idx:end_idx, "is_transition_window"] = 1.0
            
        # Chronological Split
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Scale Features
        scaler_moe = MinMaxScaler()
        train_df[moe_features] = scaler_moe.fit_transform(train_df[moe_features])
        test_df[moe_features] = scaler_moe.transform(test_df[moe_features])
        
        for h in horizons:
            print(f"Horizon: {h} days")
            target_col = f"target_res_return_{h}d"
            
            # Load target scaler saved in train_baselines.py
            scaler_target_file = os.path.join(models_dir, f"{ticker}_scaler_target_h{h}.pkl")
            if not os.path.exists(scaler_target_file):
                print(f"  Warning: Target scaler {scaler_target_file} not found.")
                continue
            scaler_target = joblib.load(scaler_target_file)
            
            # Drop target rows with NaNs
            train_df_h = train_df.dropna(subset=[target_col]).copy()
            test_df_h = test_df.dropna(subset=[target_col]).copy()
            
            if len(train_df_h) < lookback or len(test_df_h) < lookback:
                print(f"  Warning: Not enough data for horizon {h}d.")
                continue
                
            train_df_h["target_scaled"] = scaler_target.transform(train_df_h[[target_col]])
            test_df_h["target_scaled"] = scaler_target.transform(test_df_h[[target_col]])
            
            # Create sequences with weights
            X_train, r_train, y_train, w_train = create_sequences_moe_v2(train_df_h, moe_features, target_col="target_scaled", lookback=lookback, horizon=h)
            X_test, r_test, y_test, w_test = create_sequences_moe_v2(test_df_h, moe_features, target_col="target_scaled", lookback=lookback, horizon=h)
            
            actual_y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # Train Weighted MoE (Model 4 v2)
            print(f"  Training Weighted MoE Model (transition count: {np.sum(w_train == 2.0)})...")
            train_ds = StockSequenceMoeDatasetV2(X_train, r_train, y_train, w_train)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            moe_model = RegimeConditionedMoE(input_size=len(moe_features), sentiment_feature_idx=sentiment_idx)
            moe_model = train_moe_model_v2(moe_model, train_loader, epochs=epochs, lr=learning_rate)
            
            # Save weighted MoE weights as _moe_v2_h{h}.pt
            model_path = os.path.join(models_dir, f"{ticker}_moe_v2_h{h}.pt")
            torch.save(moe_model.state_dict(), model_path)
            print(f"  Saved weights to {model_path}")
            
            # Predict and evaluate
            moe_model.eval()
            with torch.no_grad():
                preds_moe_scaled = moe_model(
                    torch.tensor(X_test, dtype=torch.float32),
                    torch.tensor(r_test, dtype=torch.float32)
                ).squeeze().numpy()
                
            preds_moe = scaler_target.inverse_transform(preds_moe_scaled.reshape(-1, 1)).flatten()
            
            rmse_moe = np.sqrt(mean_squared_error(actual_y_test, preds_moe))
            mae_moe = mean_absolute_error(actual_y_test, preds_moe)
            mape_moe = calculate_mape(actual_y_test, preds_moe)
            
            print(f"  MoE v2: RMSE={rmse_moe:.5f} | MAE={mae_moe:.5f} | MAPE={mape_moe:.2f}%")
            
            # Save to run_log_v2.csv
            new_log = pd.DataFrame([{
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "ticker": ticker,
                "horizon": f"{h}d",
                "model": "Weighted_Regime_MoE",
                "rmse": rmse_moe,
                "mae": mae_moe,
                "mape": mape_moe
            }])
            new_log.to_csv(log_file, mode="a", header=False, index=False)

if __name__ == "__main__":
    main()
