import torch
import torch.nn as nn

class RegimeConditionedMoE(nn.Module):
    def __init__(self, input_size, sentiment_feature_idx=-1, hidden_size=64, num_layers=2, output_size=1):
        super(RegimeConditionedMoE, self).__init__()
        self.sentiment_feature_idx = sentiment_feature_idx
        
        # Shared LSTM Backbone to extract temporal representations
        self.backbone = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # 3 Expert Heads specializing in Bullish (0), Bearish (1), and Sideways (2) regimes
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
        # Rule-based Sentiment Scaling by Regime:
        # Higher weight in Sideways (1.5), lower in Bullish (0.5) and Bearish (0.2)
        # We blend the scaling factors using HMM state probabilities
        # regime_probs columns correspond to: 0 -> Bull, 1 -> Bear, 2 -> Side
        batch_size, seq_len, _ = x.shape
        
        # Extract probabilities
        prob_bull = regime_probs[:, 0]
        prob_bear = regime_probs[:, 1]
        prob_side = regime_probs[:, 2]
        
        # Dynamic scale for sentiment score column
        sent_scale = prob_bull * 0.5 + prob_bear * 0.2 + prob_side * 1.5 # shape: (batch_size,)
        
        # Reshape for multiplication: (batch_size, 1) to match (batch_size, seq_len)
        sent_scale_unsqueezed = sent_scale.unsqueeze(-1)
        
        # Clone input to avoid in-place modification
        x_scaled = x.clone()
        if self.sentiment_feature_idx is not None:
            # Scale the sentiment feature column
            x_scaled[:, :, self.sentiment_feature_idx] = x[:, :, self.sentiment_feature_idx] * sent_scale_unsqueezed
            
        # 1. Pass through Shared Backbone
        features, _ = self.backbone(x_scaled)
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
