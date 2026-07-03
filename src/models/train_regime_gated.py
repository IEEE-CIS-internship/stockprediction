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

class StockSequenceMoeDataset(Dataset):
    def __init__(self, X, regime_probs, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.regime_probs = torch.tensor(regime_probs, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.regime_probs[idx], self.y[idx]

def create_sequences_moe(df, features, target_col, lookback=30, horizon=1):
    X, regime_probs, y = [], [], []
    feature_data = df[features].values
    regime_data = df[["prob_bull", "prob_bear", "prob_side"]].values
    target_data = df[target_col].values
    
    # Target index is t + horizon - 1 + lookback
    for i in range(len(df) - lookback - horizon + 1):
        X.append(feature_data[i : i + lookback])
        # Gating probabilities are at the end of the sequence (t + lookback - 1)
        regime_probs.append(regime_data[i + lookback - 1])
        # Target is at i + lookback + horizon - 1
        y.append(target_data[i + lookback - 1 + horizon])
        
    return np.array(X), np.array(regime_probs), np.array(y)

def train_moe_model(model, dataloader, epochs=20, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for batch_x, batch_regime, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x, batch_regime)
            loss = criterion(predictions.squeeze(), batch_y)
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
    parser = argparse.ArgumentParser(description="Train Regime-Gated MoE Model")
    parser.add_argument("--in", dest="in_dir", required=True, help="Features directory")
    parser.add_argument("--out", nargs=2, required=True, help="Output directories: [exp_dir, tab_dir]")
    
    args = parser.parse_args()
    exp_dir, tab_dir = args.out[0], args.out[1]
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Feature list for MoE: Technical + news sentiment + macro variables + corporate events
    moe_features = [
        "open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30",
        "sentiment_score", "vix", "crude", "usdinr", "is_split_day", "is_dividend_day", "days_to_earnings"
    ]
    # sentiment score index in moe_features to pass to the model for scaling
    sentiment_idx = moe_features.index("sentiment_score")
    
    # Horizons to train: 1, 3, 7 days
    horizons = [1, 3, 7]
    lookback = 30
    
    # Hyperparameters
    epochs = 20
    batch_size = 32
    learning_rate = 0.001
    
    # Setup Experiments Log
    log_file = os.path.join(exp_dir, "run_log.csv")
    
    # Find all stock features files
    tickers = [f.split("_features.csv")[0] for f in os.listdir(args.in_dir) if f.endswith("_features.csv")]
    
    for ticker in tickers:
        print(f"\n--- Training Regime-Gated MoE for {ticker} ---")
        
        # Load features
        features_file = os.path.join(args.in_dir, f"{ticker}_features.csv")
        df = pd.read_csv(features_file)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Chronological Split (80% train, 20% test)
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()
        
        # Scale Features
        scaler_moe = MinMaxScaler()
        train_df[moe_features] = scaler_moe.fit_transform(train_df[moe_features])
        test_df[moe_features] = scaler_moe.transform(test_df[moe_features])
        
        # Save input scaler for MoE
        joblib.dump(scaler_moe, os.path.join(models_dir, f"{ticker}_scaler_moe.pkl"))
        
        for h in horizons:
            print(f"Horizon: {h} days")
            target_col = f"target_res_return_{h}d"
            
            # Load target scaler saved in train_baselines.py
            scaler_target_file = os.path.join(models_dir, f"{ticker}_scaler_target_h{h}.pkl")
            if not os.path.exists(scaler_target_file):
                print(f"  Warning: Target scaler {scaler_target_file} not found. Ensure train_baselines.py is run first.")
                continue
            scaler_target = joblib.load(scaler_target_file)
            
            # Drop target rows with NaNs
            train_df_h = train_df.dropna(subset=[target_col]).copy()
            test_df_h = test_df.dropna(subset=[target_col]).copy()
            
            if len(train_df_h) < lookback or len(test_df_h) < lookback:
                print(f"  Warning: Not enough data to train horizon {h}d for {ticker}.")
                continue
                
            train_df_h["target_scaled"] = scaler_target.transform(train_df_h[[target_col]])
            test_df_h["target_scaled"] = scaler_target.transform(test_df_h[[target_col]])
            
            # Create sequences
            X_train, r_train, y_train = create_sequences_moe(train_df_h, moe_features, target_col="target_scaled", lookback=lookback, horizon=h)
            X_test, r_test, y_test = create_sequences_moe(test_df_h, moe_features, target_col="target_scaled", lookback=lookback, horizon=h)
            
            actual_y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # -------------------------------------------------------------
            # Model 4: Regime-Conditioned MoE
            # -------------------------------------------------------------
            print("  Training MoE Model...")
            train_ds = StockSequenceMoeDataset(X_train, r_train, y_train)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            
            moe_model = RegimeConditionedMoE(input_size=len(moe_features), sentiment_feature_idx=sentiment_idx)
            moe_model = train_moe_model(moe_model, train_loader, epochs=epochs, lr=learning_rate)
            
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
            
            # Save MoE model checkpoint
            torch.save(moe_model.state_dict(), os.path.join(models_dir, f"{ticker}_moe_h{h}.pt"))
            
            # -------------------------------------------------------------
            # Log Results to experiments/run_log.csv
            # -------------------------------------------------------------
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entry = {
                "timestamp": timestamp,
                "ticker": ticker,
                "horizon": f"{h}d",
                "model": "Regime_MoE",
                "loss_type": "MSE",
                "train_loss": 0.0,
                "test_rmse": rmse_moe,
                "test_mae": mae_moe,
                "test_mape": mape_moe
            }
            df_log_append = pd.DataFrame([log_entry])
            df_log_append.to_csv(log_file, mode='a', header=False, index=False)
            
            print(f"  Regime MoE Model -> RMSE: {rmse_moe:.4f} | MAE: {mae_moe:.4f} | MAPE: {mape_moe:.2f}%")
            
            # Verify gating behavior: print average weights for the test set
            mean_weights = np.mean(r_test, axis=0)
            print(f"    Test set average expert routing weights: Bullish={mean_weights[0]*100:.1f}%, Bearish={mean_weights[1]*100:.1f}%, Sideways={mean_weights[2]*100:.1f}%")
            
    print("\nRegime-gated model training complete! Results logged to:", log_file)

if __name__ == "__main__":
    main()
