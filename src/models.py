import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression

# 1. Baseline Linear Regression
class BaselineLinearRegression:
    def __init__(self):
        self.model = LinearRegression()
        
    def fit(self, X, y):
        # Expects flattened 2D arrays: (N, num_features * seq_len)
        self.model.fit(X, y)
        
    def predict(self, X):
        return self.model.predict(X)

# 2. Plain LSTM Model (Standard Sequence Predictor)
class PlainLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(PlainLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 3. LSTM + Sentiment Model (same architecture, just takes more input features)
class LSTMSentiment(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(LSTMSentiment, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# 4. Regime-Conditioned Mixture-of-Experts (MoE) Forecaster
class RegimeConditionedMoE(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1):
        super(RegimeConditionedMoE, self).__init__()
        
        # Shared Backbone to extract temporal representations
        self.backbone = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # 3 Expert Heads specializing in Bullish, Bearish, and Sideways regimes
        self.expert_bull = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        self.expert_bear = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        self.expert_side = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )
        
    def forward(self, x, regime_probs):
        """
        x shape: (batch_size, seq_len, input_size)
        regime_probs shape: (batch_size, 3) -> probability weights for [Bull, Bear, Side]
        """
        # 1. Pass through Shared Backbone
        features, _ = self.backbone(x)
        features = features[:, -1, :]  # Last time step output: (batch_size, hidden_size)
        
        # 2. Get predictions from each expert
        pred_bull = self.expert_bull(features)  # (batch_size, output_size)
        pred_bear = self.expert_bear(features)  # (batch_size, output_size)
        pred_side = self.expert_side(features)  # (batch_size, output_size)
        
        # 3. Stack expert outputs: shape (batch_size, 3, output_size)
        expert_outputs = torch.stack([pred_bull, pred_bear, pred_side], dim=1)
        
        # 4. Gating / Weighting by HMM regime probabilities
        # regime_probs shape (batch_size, 3) -> reshape to (batch_size, 3, 1) for broadcasting
        weights = regime_probs.unsqueeze(-1)
        
        # Perform weighted sum over the experts dimension (dim=1)
        gated_output = torch.sum(expert_outputs * weights, dim=1)  # (batch_size, output_size)
        
        return gated_output
