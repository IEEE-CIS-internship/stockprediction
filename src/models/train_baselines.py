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

# Import model architectures
from lstm import PlainLSTM
from lstm_sentiment import LSTMSentiment
from linear import BaselineLinearRegression

# Define linear model wrapper in case it's not present in linear.py
# The baseline models are scikit-learn LinearRegression models.
# Let's write a simple class wrapper or use sklearn directly.

class StockSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def create_sequences(df, features, target_col, lookback=30, horizon=1):
    X, y = [], []
    feature_data = df[features].values
    target_data = df[target_col].values
    
    # Target index is t + horizon - 1 + lookback
    # The last row we can use to predict is: len(df) - lookback - horizon + 1
    for i in range(len(df) - lookback - horizon + 1):
        X.append(feature_data[i : i + lookback])
        # Target is at i + lookback + horizon - 1
        y.append(target_data[i + lookback - 1 + horizon])
        
    return np.array(X), np.array(y)

def train_pytorch_model(model, dataloader, epochs=15, lr=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            predictions = model(batch_x)
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
    parser = argparse.ArgumentParser(description="Train Baseline Models")
    parser.add_argument("--in", dest="in_dir", required=True, help="Features directory")
    parser.add_argument("--out", nargs=2, required=True, help="Output directories: [exp_dir, tab_dir]")
    
    args = parser.parse_args()
    exp_dir, tab_dir = args.out[0], args.out[1]
    
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Feature lists
    tech_features = ["open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30"]
    all_features = tech_features + ["sentiment_score"]
    
    # Horizons to train: 1, 3, 7 days
    horizons = [1, 3, 7]
    lookback = 30
    
    # Find all stock features files
    tickers = [f.split("_features.csv")[0] for f in os.listdir(args.in_dir) if f.endswith("_features.csv")]
    
    # Hyperparameters
    epochs = 15
    batch_size = 32
    learning_rate = 0.001
    
    # Setup Experiments Log
    log_file = os.path.join(exp_dir, "run_log.csv")
    if not os.path.exists(log_file):
        df_log = pd.DataFrame(columns=["timestamp", "ticker", "horizon", "model", "loss_type", "train_loss", "test_rmse", "test_mae", "test_mape"])
        df_log.to_csv(log_file, index=False)
    
    for ticker in tickers:
        print(f"\n--- Training baselines for {ticker} ---")
        
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
        scaler_tech = MinMaxScaler()
        scaler_all = MinMaxScaler()
        
        # Scale inputs (inplace scaling)
        train_df[tech_features] = scaler_tech.fit_transform(train_df[tech_features])
        test_df[tech_features] = scaler_tech.transform(test_df[tech_features])
        
        train_df[all_features] = scaler_all.fit_transform(train_df[all_features])
        test_df[all_features] = scaler_all.transform(test_df[all_features])
        
        # Save input scalers
        joblib.dump(scaler_tech, os.path.join(models_dir, f"{ticker}_scaler_tech.pkl"))
        joblib.dump(scaler_all, os.path.join(models_dir, f"{ticker}_scaler_all.pkl"))
        
        for h in horizons:
            print(f"Horizon: {h} days")
            target_col = f"target_res_return_{h}d"
            
            # Drop target rows with NaNs (which occur at the end due to shift(-h))
            # We must drop target NaNs separately for each horizon
            train_df_h = train_df.dropna(subset=[target_col]).copy()
            test_df_h = test_df.dropna(subset=[target_col]).copy()
            
            if len(train_df_h) < lookback or len(test_df_h) < lookback:
                print(f"  Warning: Not enough data to train horizon {h}d for {ticker}.")
                continue
                
            # Fit target scaler (to scale residual returns target)
            scaler_target = MinMaxScaler()
            train_df_h["target_scaled"] = scaler_target.fit_transform(train_df_h[[target_col]])
            test_df_h["target_scaled"] = scaler_target.transform(test_df_h[[target_col]])
            
            # Save target scaler specific to this horizon
            joblib.dump(scaler_target, os.path.join(models_dir, f"{ticker}_scaler_target_h{h}.pkl"))
            
            # Create sequences
            # Model 1 & 2: Technical features
            X_train_tech, y_train = create_sequences(train_df_h, tech_features, target_col="target_scaled", lookback=lookback, horizon=h)
            X_test_tech, y_test = create_sequences(test_df_h, tech_features, target_col="target_scaled", lookback=lookback, horizon=h)
            
            # Model 3: All features
            X_train_all, _ = create_sequences(train_df_h, all_features, target_col="target_scaled", lookback=lookback, horizon=h)
            X_test_all, _ = create_sequences(test_df_h, all_features, target_col="target_scaled", lookback=lookback, horizon=h)
            
            actual_y_test = scaler_target.inverse_transform(y_test.reshape(-1, 1)).flatten()
            
            # -------------------------------------------------------------
            # Model 1: Baseline Linear Regression
            # -------------------------------------------------------------
            print("  Training Linear Regression...")
            # Flatten sequence input to 2D
            X_train_flat = X_train_tech.reshape(len(X_train_tech), -1)
            X_test_flat = X_test_tech.reshape(len(X_test_tech), -1)
            
            lr_model = BaselineLinearRegression()
            lr_model.fit(X_train_flat, y_train)
            
            # Predict and evaluate
            preds_lr_scaled = lr_model.predict(X_test_flat)
            preds_lr = scaler_target.inverse_transform(preds_lr_scaled.reshape(-1, 1)).flatten()
            
            rmse_lr = np.sqrt(mean_squared_error(actual_y_test, preds_lr))
            mae_lr = mean_absolute_error(actual_y_test, preds_lr)
            mape_lr = calculate_mape(actual_y_test, preds_lr)
            
            # Save LR model
            joblib.dump(lr_model, os.path.join(models_dir, f"{ticker}_linear_h{h}.pkl"))
            
            # -------------------------------------------------------------
            # Model 2: Plain LSTM
            # -------------------------------------------------------------
            print("  Training Plain LSTM...")
            train_ds_plain = StockSequenceDataset(X_train_tech, y_train)
            train_loader_plain = DataLoader(train_ds_plain, batch_size=batch_size, shuffle=True)
            
            plain_lstm = PlainLSTM(input_size=len(tech_features))
            plain_lstm = train_pytorch_model(plain_lstm, train_loader_plain, epochs=epochs, lr=learning_rate)
            
            # Predict and evaluate
            plain_lstm.eval()
            with torch.no_grad():
                preds_plain_scaled = plain_lstm(torch.tensor(X_test_tech, dtype=torch.float32)).squeeze().numpy()
            
            preds_plain = scaler_target.inverse_transform(preds_plain_scaled.reshape(-1, 1)).flatten()
            
            rmse_plain = np.sqrt(mean_squared_error(actual_y_test, preds_plain))
            mae_plain = mean_absolute_error(actual_y_test, preds_plain)
            mape_plain = calculate_mape(actual_y_test, preds_plain)
            
            # Save Plain LSTM model
            torch.save(plain_lstm.state_dict(), os.path.join(models_dir, f"{ticker}_plain_lstm_h{h}.pt"))
            
            # -------------------------------------------------------------
            # Model 3: LSTM + Sentiment
            # -------------------------------------------------------------
            print("  Training LSTM + Sentiment...")
            train_ds_sent = StockSequenceDataset(X_train_all, y_train)
            train_loader_sent = DataLoader(train_ds_sent, batch_size=batch_size, shuffle=True)
            
            sent_lstm = LSTMSentiment(input_size=len(all_features))
            sent_lstm = train_pytorch_model(sent_lstm, train_loader_sent, epochs=epochs, lr=learning_rate)
            
            # Predict and evaluate
            sent_lstm.eval()
            with torch.no_grad():
                preds_sent_scaled = sent_lstm(torch.tensor(X_test_all, dtype=torch.float32)).squeeze().numpy()
                
            preds_sent = scaler_target.inverse_transform(preds_sent_scaled.reshape(-1, 1)).flatten()
            
            rmse_sent = np.sqrt(mean_squared_error(actual_y_test, preds_sent))
            mae_sent = mean_absolute_error(actual_y_test, preds_sent)
            mape_sent = calculate_mape(actual_y_test, preds_sent)
            
            # Save LSTM + Sentiment model
            torch.save(sent_lstm.state_dict(), os.path.join(models_dir, f"{ticker}_sent_lstm_h{h}.pt"))
            
            # -------------------------------------------------------------
            # Log Results to experiments/run_log.csv
            # -------------------------------------------------------------
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            log_entries = [
                {"timestamp": timestamp, "ticker": ticker, "horizon": f"{h}d", "model": "Linear_Regression", "loss_type": "MSE", "train_loss": 0.0, "test_rmse": rmse_lr, "test_mae": mae_lr, "test_mape": mape_lr},
                {"timestamp": timestamp, "ticker": ticker, "horizon": f"{h}d", "model": "Plain_LSTM", "loss_type": "MSE", "train_loss": 0.0, "test_rmse": rmse_plain, "test_mae": mae_plain, "test_mape": mape_plain},
                {"timestamp": timestamp, "ticker": ticker, "horizon": f"{h}d", "model": "LSTM_Sentiment", "loss_type": "MSE", "train_loss": 0.0, "test_rmse": rmse_sent, "test_mae": mae_sent, "test_mape": mape_sent}
            ]
            df_log_append = pd.DataFrame(log_entries)
            df_log_append.to_csv(log_file, mode='a', header=False, index=False)
            
            print(f"  Linear Regression -> RMSE: {rmse_lr:.4f} | MAE: {mae_lr:.4f} | MAPE: {mape_lr:.2f}%")
            print(f"  Plain LSTM        -> RMSE: {rmse_plain:.4f} | MAE: {mae_plain:.4f} | MAPE: {mape_plain:.2f}%")
            print(f"  LSTM + Sentiment  -> RMSE: {rmse_sent:.4f} | MAE: {mae_sent:.4f} | MAPE: {mape_sent:.2f}%")
            
    print("\nTraining of baselines complete! Results logged to:", log_file)

if __name__ == "__main__":
    main()
