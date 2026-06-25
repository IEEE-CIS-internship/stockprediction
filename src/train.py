import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from models import BaselineLinearRegression, PlainLSTM, LSTMSentiment, RegimeConditionedMoE

class StockSequenceDataset(Dataset):
    def __init__(self, X, y, regime_probs=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.regime_probs = torch.tensor(regime_probs, dtype=torch.float32) if regime_probs is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.regime_probs is not None:
            return self.X[idx], self.regime_probs[idx], self.y[idx]
        return self.X[idx], self.y[idx]

def create_sequences(df, features, target_col="close", lookback=30, horizon=1):
    """
    Creates sequences of length `lookback` and targets `horizon` days ahead.
    """
    X, y, regime_probs = [], [], []
    
    # We need the values of the features and the target
    feature_data = df[features].values
    target_data = df[target_col].values
    
    # Extract HMM probabilities for gating
    has_regime = "prob_bull" in df.columns
    if has_regime:
        regime_data = df[["prob_bull", "prob_bear", "prob_side"]].values
        
    for i in range(len(df) - lookback - horizon + 1):
        # Sequence of features from t to t + lookback - 1
        X.append(feature_data[i : i + lookback])
        # Target at t + lookback - 1 + horizon
        y.append(target_data[i + lookback - 1 + horizon])
        # Gate probabilities at the end of the sequence (t + lookback - 1)
        if has_regime:
            regime_probs.append(regime_data[i + lookback - 1])
            
    return np.array(X), np.array(y), (np.array(regime_probs) if has_regime else None)

def train_pytorch_model(model, dataloader, epochs=20, lr=0.001, is_moe=False):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            
            if is_moe:
                batch_x, batch_regime, batch_y = batch
                predictions = model(batch_x, batch_regime)
            else:
                batch_x, batch_y = batch
                predictions = model(batch_x)
                
            loss = criterion(predictions.squeeze(), batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_y)
            
        # Optional print
        # print(f"Epoch {epoch+1}/{epochs} Loss: {epoch_loss / len(dataloader.dataset):.4f}")
    return model

def run_training_pipeline(ticker_name, start_date="2020-01-01", end_date="2026-06-25"):
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_filepath = os.path.join(base_dir, "data", f"regime_{ticker_name}_processed.csv")
    models_dir = os.path.join(base_dir, "models")
    
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        
    if not os.path.exists(data_filepath):
        print(f"Data not found for {ticker_name} at {data_filepath}. Please run pipeline and regime classifier first.")
        return
        
    df = pd.read_csv(data_filepath)
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    
    # Handle missing news sentiment score if news fetching is sparse
    if "sentiment_score" not in df.columns:
        # Mock sentiment score for structural completeness if not scraped
        df["sentiment_score"] = np.random.uniform(-0.5, 0.5, len(df))
        
    # Technical Features list
    tech_features = ["open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30"]
    all_features = tech_features + ["sentiment_score"]
    
    # Horizons to train: 1, 3, 7
    horizons = [1, 3, 7]
    lookback = 30
    
    # Chronological Split (80% train, 20% test)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    
    # Scale Features
    scaler_tech = MinMaxScaler()
    scaler_all = MinMaxScaler()
    
    train_df[tech_features] = scaler_tech.fit_transform(train_df[tech_features])
    test_df[tech_features] = scaler_tech.transform(test_df[tech_features])
    
    train_df[all_features] = scaler_all.fit_transform(train_df[all_features])
    test_df[all_features] = scaler_all.transform(test_df[all_features])
    
    # Fit target scaler to invert predictions later
    target_scaler = MinMaxScaler()
    train_df["close_scaled"] = target_scaler.fit_transform(train_df[["close"]])
    test_df["close_scaled"] = target_scaler.transform(test_df[["close"]])
    
    results = {}
    
    for h in horizons:
        print(f"Training models for Horizon {h} days...")
        
        # 1. Prepare sequence datasets
        # A. Technical features only (for Plain LSTM)
        X_train_tech, y_train, _ = create_sequences(train_df, tech_features, target_col="close_scaled", lookback=lookback, horizon=h)
        X_test_tech, y_test, _ = create_sequences(test_df, tech_features, target_col="close_scaled", lookback=lookback, horizon=h)
        
        # B. All features including sentiment & regime probabilities (for LSTM+Sent and MoE)
        X_train_all, _, r_train = create_sequences(train_df, all_features, target_col="close_scaled", lookback=lookback, horizon=h)
        X_test_all, _, r_test = create_sequences(test_df, all_features, target_col="close_scaled", lookback=lookback, horizon=h)
        
        # -------------------------------------------------------------
        # Model 1: Baseline Linear Regression
        # -------------------------------------------------------------
        # Flatten sequence input for 2D Linear Regression
        X_train_flat = X_train_tech.reshape(len(X_train_tech), -1)
        lr_model = BaselineLinearRegression()
        lr_model.fit(X_train_flat, y_train)
        
        # -------------------------------------------------------------
        # Model 2: Plain LSTM
        # -------------------------------------------------------------
        train_ds_plain = StockSequenceDataset(X_train_tech, y_train)
        train_loader_plain = DataLoader(train_ds_plain, batch_size=32, shuffle=True)
        plain_lstm = PlainLSTM(input_size=len(tech_features))
        plain_lstm = train_pytorch_model(plain_lstm, train_loader_plain, epochs=15, is_moe=False)
        
        # -------------------------------------------------------------
        # Model 3: LSTM + Sentiment
        # -------------------------------------------------------------
        train_ds_sent = StockSequenceDataset(X_train_all, y_train)
        train_loader_sent = DataLoader(train_ds_sent, batch_size=32, shuffle=True)
        sent_lstm = LSTMSentiment(input_size=len(all_features))
        sent_lstm = train_pytorch_model(sent_lstm, train_loader_sent, epochs=15, is_moe=False)
        
        # -------------------------------------------------------------
        # Model 4: Regime-Conditioned MoE
        # -------------------------------------------------------------
        train_ds_moe = StockSequenceDataset(X_train_all, y_train, r_train)
        train_loader_moe = DataLoader(train_ds_moe, batch_size=32, shuffle=True)
        moe_model = RegimeConditionedMoE(input_size=len(all_features))
        moe_model = train_pytorch_model(moe_model, train_loader_moe, epochs=20, is_moe=True)
        
        # Save checkpoints
        torch.save(plain_lstm.state_dict(), os.path.join(models_dir, f"{ticker_name}_plain_lstm_h{h}.pt"))
        torch.save(sent_lstm.state_dict(), os.path.join(models_dir, f"{ticker_name}_sent_lstm_h{h}.pt"))
        torch.save(moe_model.state_dict(), os.path.join(models_dir, f"{ticker_name}_moe_h{h}.pt"))
        
        # Save Scalers for inference
        import joblib
        joblib.dump(scaler_tech, os.path.join(models_dir, f"{ticker_name}_scaler_tech.pkl"))
        joblib.dump(scaler_all, os.path.join(models_dir, f"{ticker_name}_scaler_all.pkl"))
        joblib.dump(target_scaler, os.path.join(models_dir, f"{ticker_name}_scaler_target.pkl"))
        
        print(f"Saved all models for Horizon {h} days successfully.")
        
if __name__ == "__main__":
    # Test on a stock if data exists
    run_training_pipeline("ICICIBANK")
