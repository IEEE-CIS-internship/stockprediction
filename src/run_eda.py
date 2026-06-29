import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for professional-looking plots
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

def load_data(in_dir, ticker):
    price_file = os.path.join(in_dir, f"{ticker}_prices.csv")
    sent_file = os.path.join(in_dir, f"{ticker}_sentiment.csv")
    events_file = os.path.join(in_dir, f"{ticker}_events.csv")
    
    if not os.path.exists(price_file):
        return None, None, None
        
    df_prices = pd.read_csv(price_file)
    df_prices["date"] = pd.to_datetime(df_prices["date"])
    
    df_sent = pd.read_csv(sent_file) if os.path.exists(sent_file) else pd.DataFrame(columns=["date", "sentiment_score"])
    if not df_sent.empty:
        df_sent["date"] = pd.to_datetime(df_sent["date"])
        
    df_events = pd.read_csv(events_file) if os.path.exists(events_file) else pd.DataFrame(columns=["date", "stock_splits", "dividends", "is_earnings_day"])
    if not df_events.empty:
        df_events["date"] = pd.to_datetime(df_events["date"])
        
    return df_prices, df_sent, df_events

def main():
    parser = argparse.ArgumentParser(description="AuraTrade AI Exploratory Data Analysis")
    parser.add_argument("--in", dest="in_dir", required=True, help="Input raw directory")
    parser.add_argument("--out", nargs=2, required=True, help="Output directories: [figures_dir, tables_dir]")
    
    args = parser.parse_args()
    fig_dir, tab_dir = args.out[0], args.out[1]
    
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    
    # Load index prices
    index_file = os.path.join(args.in_dir, "index_prices.csv")
    if not os.path.exists(index_file):
        print("CRITICAL: index_prices.csv not found in raw directory.")
        return
    df_index = pd.read_csv(index_file)
    df_index["date"] = pd.to_datetime(df_index["date"])
    df_index.sort_values("date", inplace=True)
    
    # Load macro variables
    macro_file = os.path.join(args.in_dir, "macro_variables.csv")
    df_macro = pd.read_csv(macro_file) if os.path.exists(macro_file) else pd.DataFrame()
    if not df_macro.empty:
        df_macro["date"] = pd.to_datetime(df_macro["date"])
        df_macro.sort_values("date", inplace=True)
        
    # Get all tickers from filenames
    tickers = [f.split("_prices.csv")[0] for f in os.listdir(args.in_dir) if f.endswith("_prices.csv")]
    
    print(f"Running EDA on tickers: {tickers}")
    
    quality_audit = []
    
    for ticker in tickers:
        print(f"Analyzing {ticker}...")
        df_prices, df_sent, df_events = load_data(args.in_dir, ticker)
        if df_prices is None:
            continue
            
        df_prices.sort_values("date", inplace=True)
        df_prices["returns"] = df_prices["close"].pct_change()
        
        # 1. Price + Volume Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
        ax1.plot(df_prices["date"], df_prices["close"], color="#17becf", label=f"{ticker} Close", linewidth=1.5)
        ax1.set_title(f"{ticker} Historical Close Price and Volume (5 Years)", color="white", fontsize=12)
        ax1.set_ylabel("Close Price (INR)", color="white")
        ax1.legend(loc="upper left", facecolor="#151722", labelcolor="white")
        
        ax2.bar(df_prices["date"], df_prices["volume"], color="#7f7f7f", alpha=0.6, label="Volume")
        ax2.set_ylabel("Volume", color="white")
        ax2.set_xlabel("Date", color="white")
        ax2.legend(loc="upper left", facecolor="#151722", labelcolor="white")
        
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{ticker}_price_volume.png"), dpi=150)
        plt.close(fig)
        
        # Merge prices with index returns to calculate residual returns
        df_aligned = pd.merge(df_prices[["date", "close", "returns"]], df_index[["date", "close"]], on="date", suffixes=("", "_nifty"))
        df_aligned["returns_nifty"] = df_aligned["close_nifty"].pct_change()
        
        # Multi-horizon residual returns
        df_aligned["res_return_1d"] = df_aligned["returns"] - df_aligned["returns_nifty"]
        
        # 3-day and 7-day returns
        df_aligned["returns_3d"] = df_aligned["close"].pct_change(3)
        df_aligned["returns_nifty_3d"] = df_aligned["close_nifty"].pct_change(3)
        df_aligned["res_return_3d"] = df_aligned["returns_3d"] - df_aligned["returns_nifty_3d"]
        
        df_aligned["returns_7d"] = df_aligned["close"].pct_change(7)
        df_aligned["returns_nifty_7d"] = df_aligned["close_nifty"].pct_change(7)
        df_aligned["res_return_7d"] = df_aligned["returns_7d"] - df_aligned["returns_nifty_7d"]
        
        # 2. Residual Return Distribution Plot
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(df_aligned["res_return_1d"].dropna(), kde=True, color="#bcbd22", bins=100, ax=ax)
        # Calculate skew and kurtosis
        skew = df_aligned["res_return_1d"].skew()
        kurt = df_aligned["res_return_1d"].kurt()
        ax.set_title(f"{ticker} 1-Day Residual Return Distribution\nSkewness: {skew:.2f} | Kurtosis: {kurt:.2f} (Fat Tails)", color="white", fontsize=11)
        ax.set_xlabel("Residual Return", color="white")
        ax.set_ylabel("Frequency", color="white")
        fig.savefig(os.path.join(fig_dir, f"{ticker}_residual_return_dist.png"), dpi=150)
        plt.close(fig)
        
        # 3. Sentiment Score Overlay Plot
        if not df_sent.empty:
            df_merged_sent = pd.merge(df_prices, df_sent, on="date", how="left")
            df_merged_sent["sentiment_score"] = df_merged_sent["sentiment_score"].fillna(0.0)
            
            fig, ax1 = plt.subplots(figsize=(11, 5))
            ax1.plot(df_merged_sent["date"], df_merged_sent["close"], color="#17becf", label="Close Price", linewidth=1.5)
            ax1.set_ylabel("Close Price (INR)", color="white")
            ax1.set_xlabel("Date", color="white")
            
            ax2 = ax1.twinx()
            # 10-day rolling sentiment to smooth it out
            rolling_sent = df_merged_sent["sentiment_score"].rolling(window=10, min_periods=1).mean()
            ax2.fill_between(df_merged_sent["date"], rolling_sent, 0, where=(rolling_sent >= 0), color="green", alpha=0.3, label="10d Rolling Sentiment (+)")
            ax2.fill_between(df_merged_sent["date"], rolling_sent, 0, where=(rolling_sent < 0), color="red", alpha=0.3, label="10d Rolling Sentiment (-)")
            ax2.set_ylabel("Sentiment Score (smoothed)", color="white")
            ax2.set_ylim(-1.1, 1.1)
            ax2.grid(False)
            
            lines1, labels1 = ax1.get_legend_handles_labels()
            # Construct a combined legend
            plt.title(f"{ticker} Close Price & rolling Sentiment Score Overlay", color="white", fontsize=12)
            fig.savefig(os.path.join(fig_dir, f"{ticker}_sentiment_overlay.png"), dpi=150)
            plt.close(fig)
            
        # 4. Correlation Heatmap returns vs macro
        if not df_macro.empty:
            df_corr = pd.merge(df_prices[["date", "returns"]], df_macro, on="date", how="inner")
            df_corr_matrix = df_corr.drop(columns=["date"]).corr()
            
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(df_corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1.0, vmax=1.0, ax=ax, cbar_kws={'label': 'Correlation Coefficient'})
            ax.set_title(f"{ticker} Returns & Macro Factors Correlation Heatmap", color="white", fontsize=11)
            plt.tight_layout()
            fig.savefig(os.path.join(fig_dir, f"{ticker}_macro_correlation.png"), dpi=150)
            plt.close(fig)
            
        # 5. Data Quality Audit Data collection
        row_count = len(df_prices)
        date_min = df_prices["date"].min().strftime("%Y-%m-%d")
        date_max = df_prices["date"].max().strftime("%Y-%m-%d")
        null_count = df_prices.isnull().sum().sum()
        sentiment_days = len(df_sent[df_sent["sentiment_score"] != 0.0]) if not df_sent.empty else 0
        earnings_count = len(df_events[df_events["is_earnings_day"] == 1.0]) if not df_events.empty else 0
        splits_count = len(df_events[df_events["stock_splits"] > 0.0]) if not df_events.empty else 0
        
        quality_audit.append({
            "Stock": ticker,
            "Total Rows": row_count,
            "Start Date": date_min,
            "End Date": date_max,
            "Null Values": null_count,
            "Sentiment Days": sentiment_days,
            "Earnings Events": earnings_count,
            "Stock Splits": splits_count
        })

    # Save Data Quality Audit Table
    df_audit = pd.DataFrame(quality_audit)
    df_audit.to_csv(os.path.join(tab_dir, "data_quality_audit.csv"), index=False)
    
    # Save as Markdown table
    with open(os.path.join(tab_dir, "data_quality_audit.md"), "w", encoding="utf-8") as f:
        f.write("# Data Quality Audit Summary\n\n")
        f.write(df_audit.to_markdown(index=False))
        
    # 6. Macro Factors Timeline
    if not df_macro.empty:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        ax1.plot(df_macro["date"], df_macro["vix"], color="#ff7f0e", label="India VIX")
        ax1.set_ylabel("India VIX (%)", color="white")
        ax1.legend(loc="upper left", facecolor="#151722", labelcolor="white")
        ax1.set_title("Historical Exogenous Macro Factors Timeline", color="white", fontsize=12)
        
        ax2.plot(df_macro["date"], df_macro["crude"], color="#2ca02c", label="Brent Crude Price ($)")
        ax2.set_ylabel("Brent Crude", color="white")
        ax2.legend(loc="upper left", facecolor="#151722", labelcolor="white")
        
        ax3.plot(df_macro["date"], df_macro["usdinr"], color="#d62728", label="USD/INR Rate")
        ax3.set_ylabel("USD/INR", color="white")
        ax3.set_xlabel("Date", color="white")
        ax3.legend(loc="upper left", facecolor="#151722", labelcolor="white")
        
        plt.tight_layout()
        fig.savefig(os.path.join(fig_dir, "macro_factors_timeline.png"), dpi=150)
        plt.close(fig)

    print("EDA Complete! Figures saved to:", fig_dir, "| Tables saved to:", tab_dir)

if __name__ == "__main__":
    main()
