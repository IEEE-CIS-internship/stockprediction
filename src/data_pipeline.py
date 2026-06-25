import os
import pandas as pd
import yfinance as yf
import numpy as np

# Configured Ticker Symbols
TICKERS = {
    "ICICIBANK": "ICICIBANK.NS",
    "HCLTECH": "HCLTECH.NS",
    "INDIGO": "INDIGO.NS",
    "MARUTI": "MARUTI.NS",
    "TRENT": "TRENT.NS",
    "HINDALCO": "HINDALCO.NS",
    "ONGC": "ONGC.NS",
    "ADANIENT": "ADANIENT.NS"
}

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def setup_directories():
    """Create data directory if it does not exist."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        print(f"Created data directory at: {DATA_DIR}")

def download_stock_data(ticker_name, ticker_symbol, start_date="2020-01-01", end_date="2026-06-25"):
    """
    Downloads historical stock data from Yahoo Finance and calculates basic features.
    """
    print(f"Downloading data for {ticker_name} ({ticker_symbol})...")
    try:
        # Download historical data
        df = yf.download(ticker_symbol, start=start_date, end=end_date)
        if df.empty:
            print(f"Warning: No data returned for {ticker_symbol}")
            return None
        
        # Flatten MultiIndex columns if present (common in recent yfinance updates)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        df = df.reset_index()
        
        # Ensure correct naming
        df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        
        # Sort chronologically
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Feature Engineering: Technical Indicators
        # 1. Daily Returns
        df["returns"] = df["close"].pct_change()
        
        # 2. Simple Moving Averages
        df["sma_20"] = df["close"].rolling(window=20).mean()
        df["sma_50"] = df["close"].rolling(window=50).mean()
        
        # 3. Rolling Volatility (30-day standard deviation of daily returns)
        df["volatility_30"] = df["returns"].rolling(window=30).std()
        
        # Drop initial rows containing NaN due to rolling calculations
        df.dropna(subset=["sma_50", "volatility_30"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        
        # Save to csv
        filepath = os.path.join(DATA_DIR, f"{ticker_name}_processed.csv")
        df.to_csv(filepath, index=False)
        print(f"Successfully processed and saved {len(df)} records to {filepath}")
        return df
        
    except Exception as e:
        print(f"Error downloading data for {ticker_symbol}: {str(e)}")
        return None

def run_pipeline():
    """Runs data pipeline for all configured stocks."""
    setup_directories()
    for name, symbol in TICKERS.items():
        download_stock_data(name, symbol)

if __name__ == "__main__":
    run_pipeline()
