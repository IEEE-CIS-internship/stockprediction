import os
import numpy as np
import pandas as pd
import torch

class ExplainabilityEngine:
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def get_gradient_attributions(self, model, x, regime_probs=None, is_moe=False):
        """
        Calculates feature attributions using the gradients of the model's output 
        with respect to the input features (Saliency maps / Gradient attribution).
        This is fast, robust, and works for any PyTorch model without SHAP dependency issues.
        """
        model.eval()
        
        # Ensure single sequence shape (1, seq_len, num_features)
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            
        # Clone tensor and enable gradient tracking
        x_tensor = x.clone().detach().requires_grad_(True)
        
        if is_moe:
            if regime_probs is None:
                regime_probs = torch.tensor([[0.34, 0.33, 0.33]], dtype=torch.float32)
            elif len(regime_probs.shape) == 1:
                regime_probs = regime_probs.unsqueeze(0)
                
            y_pred = model(x_tensor, regime_probs)
        else:
            y_pred = model(x_tensor)
            
        # Backward pass on prediction to get gradients
        y_pred.backward()
        
        # Gradients shape: (1, seq_len, num_features)
        grads = x_tensor.grad.squeeze(0).numpy()
        
        # Sum gradients across the sequence length (temporal axis) to get feature importance
        feature_importance = np.sum(grads, axis=0)
        
        # Normalize to percentage attributions
        total_importance = np.sum(np.abs(feature_importance))
        if total_importance > 0:
            norm_attributions = (feature_importance / total_importance) * 100
        else:
            norm_attributions = np.zeros(len(self.feature_names))
            
        return {name: score for name, score in zip(self.feature_names, norm_attributions)}

    def generate_trading_summary(self, ticker_name, current_price, target_price, horizon, regime_probs, attributions):
        """
        Generates a natural-language description of the model's view and recommendations.
        """
        # Determine active regime
        regimes = ["Bullish", "Bearish", "Sideways"]
        active_idx = np.argmax(regime_probs)
        active_regime = regimes[active_idx]
        active_conf = regime_probs[active_idx] * 100
        
        # Percent change
        pct_change = ((target_price - current_price) / current_price) * 100
        direction = "upward" if pct_change >= 0 else "downward"
        action = "BUY/HOLD" if pct_change > 0.5 and active_regime != "Bearish" else "SELL/SHORT" if pct_change < -0.5 or active_regime == "Bearish" else "NEUTRAL"
        
        # Identify top positive and negative drivers from attributions
        sorted_attrs = sorted(attributions.items(), key=lambda item: item[1], reverse=True)
        top_pos = [f"{name} (+{score:.1f}%)" for name, score in sorted_attrs[:2] if score > 0]
        top_neg = [f"{name} ({score:.1f}%)" for name, score in sorted_attrs[-2:] if score < 0]
        
        summary = (
            f"**AuraTrade View for {ticker_name}:**\n"
            f"The stock is currently identified in a **{active_regime} Regime** ({active_conf:.1f}% probability). "
            f"Under this regime, the Mixture-of-Experts engine forecasts a {horizon}-day {direction} price target "
            f"of **₹{target_price:.2f}** ({pct_change:+.2f}% change from current ₹{current_price:.2f}).\n\n"
            f"**Key Drivers of this Forecast:**\n"
        )
        
        if top_pos:
            summary += f"- **Positive Forces:** {', '.join(top_pos)}\n"
        if top_neg:
            summary += f"- **Negative Drag:** {', '.join(top_neg)}\n"
            
        summary += f"\n**Swing Trading Recommendation:** **{action}**"
        return summary

if __name__ == "__main__":
    # Test generation
    engine = ExplainabilityEngine(["Returns", "Sentiment", "SMA-20", "SMA-50", "Volume"])
    mock_attrs = {"Returns": 45.2, "Sentiment": 25.1, "SMA-20": 10.4, "SMA-50": -15.2, "Volume": -4.1}
    mock_regime = [0.72, 0.08, 0.20]
    
    summary = engine.generate_trading_summary(
        ticker_name="ICICI Bank",
        current_price=1120.50,
        target_price=1140.70,
        horizon=3,
        regime_probs=mock_regime,
        attributions=mock_attrs
    )
    print(summary)
