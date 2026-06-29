import os
import argparse
import pandas as pd
import numpy as np

def calculate_days_to_next_earnings(date_series, earnings_dates):
    """
    Computes a countdown of calendar days to the next earnings announcement.
    If no next earnings date is found in the future, defaults to 90 days.
    """
    days_to_next = []
    if earnings_dates.empty:
        return [90.0] * len(date_series)
        
    earnings_dates = pd.to_datetime(earnings_dates).sort_values()
    
    for d in date_series:
        future_earnings = earnings_dates[earnings_dates > d]
        if not future_earnings.empty:
            next_earnings = future_earnings.iloc[0]
            days = (next_earnings - d).days
            days_to_next.append(float(days))
        else:
            days_to_next.append(90.0) # default fallback cycle
            
    return days_to_next

def main():
    parser = argparse.ArgumentParser(description="AuraTrade AI Feature Engineering")
    parser.add_argument("--in", dest="in_dirs", nargs=2, required=True, help="Input directories: [raw_dir, regimes_dir]")
    parser.add_argument("--out", required=True, help="Output directory for processed features")
    
    args = parser.parse_args()
    raw_dir, regimes_dir = args.in_dirs[0], args.in_dirs[1]
    os.makedirs(args.out, exist_ok=True)
    
    # Load index prices
    index_file = os.path.join(raw_dir, "index_prices.csv")
    if not os.path.exists(index_file):
        print("CRITICAL: index_prices.csv not found in raw directory.")
        return
    df_index = pd.read_csv(index_file)
    df_index["date"] = pd.to_datetime(df_index["date"])
    df_index.sort_values("date", inplace=True)
    
    # Load macro variables
    macro_file = os.path.join(raw_dir, "macro_variables.csv")
    df_macro = pd.read_csv(macro_file) if os.path.exists(macro_file) else pd.DataFrame()
    if not df_macro.empty:
        df_macro["date"] = pd.to_datetime(df_macro["date"])
        df_macro.sort_values("date", inplace=True)
        
    # Get all tickers
    tickers = [f.split("_prices.csv")[0] for f in os.listdir(raw_dir) if f.endswith("_prices.csv")]
    
    for ticker in tickers:
        print(f"Engineering features for {ticker}...")
        
        # Load raw prices
        price_file = os.path.join(raw_dir, f"{ticker}_prices.csv")
        df_stock = pd.read_csv(price_file)
        df_stock["date"] = pd.to_datetime(df_stock["date"])
        df_stock.sort_values("date", inplace=True)
        
        # Load HMM regimes
        regime_file = os.path.join(regimes_dir, f"{ticker}_regime.csv")
        if not os.path.exists(regime_file):
            print(f"  Warning: Regime file {regime_file} not found. Ensure regime_hmm.py is run first.")
            continue
        df_regime = pd.read_csv(regime_file)
        df_regime["date"] = pd.to_datetime(df_regime["date"])
        
        # Load sentiment
        sent_file = os.path.join(raw_dir, f"{ticker}_sentiment.csv")
        df_sent = pd.read_csv(sent_file) if os.path.exists(sent_file) else pd.DataFrame()
        if not df_sent.empty:
            df_sent["date"] = pd.to_datetime(df_sent["date"])
            
        # Load corporate events
        events_file = os.path.join(raw_dir, f"{ticker}_events.csv")
        df_events = pd.read_csv(events_file) if os.path.exists(events_file) else pd.DataFrame()
        if not df_events.empty:
            df_events["date"] = pd.to_datetime(df_events["date"])

        # Clean stock data
        df_stock.dropna(subset=["close"], inplace=True)
        
        # Compute stock technical indicators on raw stock prices
        df_stock["returns"] = df_stock["close"].pct_change()
        df_stock["sma_20"] = df_stock["close"].rolling(window=20).mean()
        df_stock["sma_50"] = df_stock["close"].rolling(window=50).mean()
        df_stock["volatility_30"] = df_stock["returns"].rolling(window=30).std()
        
        # Drop rows with NaNs in rolling features before merging to keep alignment clean
        df_stock.dropna(subset=["sma_50", "volatility_30"], inplace=True)
        df_stock.reset_index(drop=True, inplace=True)
        
        # Align NIFTY index close and compute index returns
        df_merged = pd.merge(df_stock, df_index[["date", "close"]], on="date", suffixes=("", "_nifty"))
        df_merged.rename(columns={"close_nifty": "close_nifty"}, inplace=True)
        
        # Compute index daily returns
        df_merged["returns_nifty"] = df_merged["close_nifty"].pct_change()
        
        # Calculate MULTI-HORIZON RESIDUAL RETURNS target variables (Shifted forward)
        # residual_return_h = forward_stock_return - forward_index_return
        for h in [1, 3, 7]:
            # Forward stock return over horizon h (pct return between t and t+h)
            df_merged[f"fwd_stock_return_{h}d"] = (df_merged["close"].shift(-h) - df_merged["close"]) / df_merged["close"]
            # Forward index return over horizon h
            df_merged[f"fwd_index_return_{h}d"] = (df_merged["close_nifty"].shift(-h) - df_merged["close_nifty"]) / df_merged["close_nifty"]
            # Forward residual return
            df_merged[f"target_res_return_{h}d"] = df_merged[f"fwd_stock_return_{h}d"] - df_merged[f"fwd_index_return_{h}d"]
            
        # Merge HMM regimes and state probabilities
        df_merged = pd.merge(df_merged, df_regime, on="date", how="left")
        
        # Merge news sentiment
        if not df_sent.empty:
            df_merged = pd.merge(df_merged, df_sent, on="date", how="left")
        else:
            df_merged["sentiment_score"] = np.nan
        df_merged["sentiment_score"] = df_merged["sentiment_score"].fillna(0.0) # Fallback to neutral
        
        # Merge macro variables
        if not df_macro.empty:
            df_merged = pd.merge(df_merged, df_macro, on="date", how="left")
            df_merged["vix"] = df_merged["vix"].ffill().bfill()
            df_merged["crude"] = df_merged["crude"].ffill().bfill()
            df_merged["usdinr"] = df_merged["usdinr"].ffill().bfill()
        else:
            df_merged["vix"] = 15.0
            df_merged["crude"] = 80.0
            df_merged["usdinr"] = 83.0
            
        # Process corporate events
        if not df_events.empty:
            # 1. splits and dividends flags
            df_merged = pd.merge(df_merged, df_events[["date", "stock_splits", "dividends"]].dropna(how="all"), on="date", how="left")
            df_merged["is_split_day"] = (df_merged["stock_splits"] > 0).astype(float)
            df_merged["is_dividend_day"] = (df_merged["dividends"] > 0).astype(float)
            df_merged.drop(columns=["stock_splits", "dividends"], inplace=True)
            
            # 2. Days to next earnings countdown
            earnings_only = df_events[df_events["is_earnings_day"] == 1.0]["date"]
            df_merged["days_to_earnings"] = calculate_days_to_next_earnings(df_merged["date"], earnings_only)
        else:
            df_merged["is_split_day"] = 0.0
            df_merged["is_dividend_day"] = 0.0
            df_merged["days_to_earnings"] = 90.0
            
        df_merged["is_split_day"] = df_merged["is_split_day"].fillna(0.0)
        df_merged["is_dividend_day"] = df_merged["is_dividend_day"].fillna(0.0)
        
        # Drop target NaNs at the end of the series due to shift(-h)
        # Save feature sets for models
        out_file = os.path.join(args.out, f"{ticker}_features.csv")
        df_merged.to_csv(out_file, index=False)
        print(f"  Final features matrix: {df_merged.shape} rows/cols. Saved to {out_file}")
        
    print("\nFeature Engineering Complete! All processed features saved to:", args.out)

if __name__ == "__main__":
    main()
