import os
import pandas as pd
import numpy as np
from hmmlearn import hmm

class RegimeClassifier:
    def __init__(self, n_states=3, random_state=42):
        self.n_states = n_states
        # Gaussian HMM for continuous observation variables
        self.model = hmm.GaussianHMM(
            n_components=n_states, 
            covariance_type="full", 
            n_iter=100, 
            random_state=random_state
        )
        self.state_map = None # Maps raw HMM states to logical states (0: Bull, 1: Bear, 2: Sideways)

    def fit_predict(self, df):
        """
        Fits the HMM on daily returns and rolling volatility,
        and returns the log-probability of states and labeled regime column.
        """
        # Exclude any rows with NaN in returns or volatility
        features = df[["returns", "volatility_30"]].dropna().values
        
        # Fit model
        self.model.fit(features)
        
        # Predict hidden states
        hidden_states = self.model.predict(features)
        
        # Get state parameters to map raw states to logical states consistently
        # Logical Mapping Strategy:
        # Bullish (State 0): Highest mean return
        # Bearish (State 1): Lowest mean return (typically high volatility)
        # Sideways (State 2): Neutral mean return (typically low volatility)
        means = self.model.means_  # Shape: (n_components, n_features)
        return_means = means[:, 0]  # First column is returns mean
        
        sorted_indices = np.argsort(return_means) # Sorts ascending (lowest to highest)
        
        # Mapping:
        # lowest return mean -> Bearish (Logical State 1)
        # highest return mean -> Bullish (Logical State 0)
        # middle return mean -> Sideways (Logical State 2)
        bear_raw = sorted_indices[0]
        side_raw = sorted_indices[1]
        bull_raw = sorted_indices[2]
        
        self.state_map = {
            bull_raw: 0,
            bear_raw: 1,
            side_raw: 2
        }
        
        # Apply mapping
        logical_states = np.array([self.state_map[s] for s in hidden_states])
        
        # Calculate state probabilities for each step
        state_probs = self.model.predict_proba(features)
        # Reorder probabilities to match logical mapping
        # state_probs has shape (N, n_components)
        logical_probs = np.zeros_like(state_probs)
        logical_probs[:, 0] = state_probs[:, bull_raw]  # Bullish
        logical_probs[:, 1] = state_probs[:, bear_raw]  # Bearish
        logical_probs[:, 2] = state_probs[:, side_raw]  # Sideways
        
        # Align lengths in case of dropna
        res_states = np.full(len(df), np.nan)
        res_states[-len(logical_states):] = logical_states
        
        # Create probability columns
        bull_prob = np.full(len(df), np.nan)
        bear_prob = np.full(len(df), np.nan)
        side_prob = np.full(len(df), np.nan)
        
        bull_prob[-len(logical_probs):] = logical_probs[:, 0]
        bear_prob[-len(logical_probs):] = logical_probs[:, 1]
        side_prob[-len(logical_probs):] = logical_probs[:, 2]
        
        df["regime"] = res_states
        df["prob_bull"] = bull_prob
        df["prob_bear"] = bear_prob
        df["prob_side"] = side_prob
        
        return df

def run_regime_classification():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    for filename in os.listdir(data_dir):
        if filename.endswith("_processed.csv") and not filename.startswith("regime_"):
            filepath = os.path.join(data_dir, filename)
            df = pd.read_csv(filepath)
            
            print(f"Running HMM regime classification on {filename}...")
            classifier = RegimeClassifier()
            df = classifier.fit_predict(df)
            
            # Save labeled data
            df.dropna(subset=["regime"], inplace=True)
            output_filepath = os.path.join(data_dir, f"regime_{filename}")
            df.to_csv(output_filepath, index=False)
            
            # Print state stats
            print(f"HMM complete. Logical State Distribution for {filename}:")
            print(df["regime"].value_counts(normalize=True))
            print("-" * 50)

if __name__ == "__main__":
    run_regime_classification()
