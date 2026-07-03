import os
import argparse
import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Professional styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "figure.facecolor": "#0f111a",
    "axes.facecolor": "#151722",
    "savefig.facecolor": "#0f111a",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#2e3039",
    "axes.edgecolor": "#2e3039",
    "font.size": 10
})

def calculate_duration_stats(df, ticker):
    """
    Calculates statistics about how long regimes persist.
    """
    regimes = df["regime"].values
    
    # Identify segment boundaries
    regime_changes = np.diff(regimes) != 0
    change_indices = np.where(regime_changes)[0] + 1
    
    # Segment lengths
    start_idx = 0
    segments = []
    
    for idx in change_indices:
        length = idx - start_idx
        regime = regimes[start_idx]
        segments.append({"regime": regime, "length": length})
        start_idx = idx
    # Append the last segment
    segments.append({"regime": regimes[start_idx], "length": len(regimes) - start_idx})
    
    df_segments = pd.DataFrame(segments)
    
    # Calculate statistics per regime
    stats = []
    regime_names = {0: "Bullish", 1: "Bearish", 2: "Sideways"}
    for r_val, name in regime_names.items():
        sub_seg = df_segments[df_segments["regime"] == r_val]
        if not sub_seg.empty:
            avg_len = sub_seg["length"].mean()
            max_len = sub_seg["length"].max()
            total_days = sub_seg["length"].sum()
            percent_time = (total_days / len(df)) * 100
        else:
            avg_len, max_len, total_days, percent_time = 0.0, 0, 0, 0.0
            
        stats.append({
            "Stock": ticker,
            "Regime": name,
            "Total Days": total_days,
            "Percentage Time (%)": f"{percent_time:.2f}%",
            "Avg Duration (days)": f"{avg_len:.1f}",
            "Max Duration (days)": max_len
        })
        
    return stats, df_segments

def main():
    parser = argparse.ArgumentParser(description="AuraTrade HMM Regime Labeller")
    parser.add_argument("--in", dest="in_dir", required=True, help="Input raw cached data directory")
    parser.add_argument("--out", required=True, help="Output directory for processed regimes")
    
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    
    # Hardcode figure output folder for easy organization
    fig_dir = "outputs/figures"
    tab_dir = "outputs/tables"
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    
    # Find all stock price files
    tickers = [f.split("_prices.csv")[0] for f in os.listdir(args.in_dir) if f.endswith("_prices.csv")]
    
    all_duration_stats = []
    
    for ticker in tickers:
        print(f"Fitting Gaussian HMM for {ticker}...")
        
        # Load stock prices
        price_file = os.path.join(args.in_dir, f"{ticker}_prices.csv")
        df = pd.read_csv(price_file)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        
        # Drop columns if they have too many NaNs
        df.dropna(subset=["close"], inplace=True)
        
        # Compute returns if not present
        if "returns" not in df.columns:
            df["returns"] = df["close"].pct_change()
            
        # Compute volatility if not present
        if "volatility_30" not in df.columns:
            df["volatility_30"] = df["returns"].rolling(window=30).std()
            
        df.dropna(subset=["returns", "volatility_30"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # SMOOTHING FEATURE ENGINEERING to prevent daily oscillating noise
        # 1. 20-day rolling mean of returns
        df["returns_mean_20"] = df["returns"].rolling(window=20).mean()
        df.dropna(subset=["returns_mean_20"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Features for HMM: [returns_mean_20, volatility_30]
        # These are smoothed inputs, preventing the model from switching state daily!
        hmm_features = df[["returns_mean_20", "volatility_30"]].values
        
        # Fit HMM
        model = hmm.GaussianHMM(
            n_components=3, 
            covariance_type="full", 
            n_iter=100, 
            random_state=42
        )
        model.fit(hmm_features)
        
        # Predict hidden states
        raw_states = model.predict(hmm_features)
        state_probs = model.predict_proba(hmm_features)
        
        # Map raw HMM states to logical states consistently
        # Logical Sorting Strategy:
        # Sort HMM components by their returns mean (model.means_[:, 0]) ascending.
        # sorted_indices[0] -> Lowest mean return -> Bearish (Logical State 1)
        # sorted_indices[1] -> Middle mean return -> Sideways (Logical State 2)
        # sorted_indices[2] -> Highest mean return -> Bullish (Logical State 0)
        means = model.means_  # Shape: (3, 2)
        return_means = means[:, 0]
        sorted_indices = np.argsort(return_means)
        
        bear_raw = sorted_indices[0]
        side_raw = sorted_indices[1]
        bull_raw = sorted_indices[2]
        
        state_map = {
            bull_raw: 0,
            bear_raw: 1,
            side_raw: 2
        }
        
        # Apply logical mapping
        logical_states = np.array([state_map[s] for s in raw_states])
        
        # Map probabilities
        logical_probs = np.zeros_like(state_probs)
        logical_probs[:, 0] = state_probs[:, bull_raw]  # Bullish
        logical_probs[:, 1] = state_probs[:, bear_raw]  # Bearish
        logical_probs[:, 2] = state_probs[:, side_raw]  # Sideways
        
        # Add columns to dataframe
        df["regime"] = logical_states
        df["prob_bull"] = logical_probs[:, 0]
        df["prob_bear"] = logical_probs[:, 1]
        df["prob_side"] = logical_probs[:, 2]
        
        # Save processed regimes
        out_file = os.path.join(args.out, f"{ticker}_regime.csv")
        df[["date", "regime", "prob_bull", "prob_bear", "prob_side"]].to_csv(out_file, index=False)
        print(f"  Saved regime labels to {out_file}")
        
        # Calculate duration statistics
        stats, df_segs = calculate_duration_stats(df, ticker)
        all_duration_stats.extend(stats)
        
        # Print transitions count to verify HMM is smoothed
        num_changes = (df["regime"].diff() != 0).sum() - 1
        print(f"  Regime transitions in dataset: {num_changes} changes (much cleaner!).")
        
        # 7. Generate Regime segment plot
        # Show price chart with colored bands
        fig, ax = plt.subplots(figsize=(12, 5))
        dates = df["date"].values
        close_prices = df["close"].values
        reg_vals = df["regime"].values
        
        ax.plot(dates, close_prices, color="white", linewidth=1.5, label="Close Price")
        
        # Shading transitions
        # Iterate and shade background
        # State colors: Bullish (0) -> Green, Bearish (1) -> Red, Sideways (2) -> Yellow
        for i in range(len(df) - 1):
            if reg_vals[i] == 0:
                color = "green"
            elif reg_vals[i] == 1:
                color = "red"
            else:
                color = "yellow"
            ax.axvspan(dates[i], dates[i+1], color=color, alpha=0.12)
            
        ax.set_title(f"{ticker} Historical Close Price with HMM Market Regimes Overlay\n(Green = Bullish, Red = Bearish, Yellow = Sideways)", color="white", fontsize=12)
        ax.set_ylabel("Price (INR)", color="white")
        ax.set_xlabel("Date", color="white")
        plt.tight_layout()
        fig_path = os.path.join(fig_dir, f"{ticker}_regime_timeline.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  Saved regime timeline chart to {fig_path}")

    # Output regime duration stats tables
    df_durations = pd.DataFrame(all_duration_stats)
    df_durations.to_csv(os.path.join(tab_dir, "regime_duration_stats.csv"), index=False)
    
    with open(os.path.join(tab_dir, "regime_duration_stats.md"), "w", encoding="utf-8") as f:
        f.write("# Market Regime Duration Statistics (HMM)\n\n")
        f.write(df_durations.to_markdown(index=False))
        
    print("\nRegime Labeling Complete! Duration stats saved to outputs/tables/")

if __name__ == "__main__":
    main()
