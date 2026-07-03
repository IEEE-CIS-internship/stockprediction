import os
import argparse
import datetime
import pandas as pd
import numpy as np
import yfinance as yf

from sentiment_analyzer import SentimentAnalyzer

# Pinned macro tickers
MACRO_TICKERS = {
    "vix": "^INDIAVIX",
    "crude": "BZ=F",       # Brent Crude Oil Futures
    "usdinr": "USDINR=X"   # USD/INR Exchange Rate
}

# Simple ticker mapping used by the Streamlit dashboard
TICKERS = {
    "ICICI Bank": "ICICIBANK.NS",
    "HCL Technologies": "HCLTECH.NS",
    "IndiGo": "INDIGO.NS",
    "Maruti Suzuki": "MARUTI.NS",
    "Trent": "TRENT.NS",
    "Hindalco": "HINDALCO.NS",
    "ONGC": "ONGC.NS",
    "Adani Enterprises": "ADANIENT.NS",
}


def download_stock_data(stock_name, ticker_symbol):
    """Download and preprocess a single stock for the dashboard app."""
    try:
        df = yf.download(ticker_symbol, period="2y", interval="1d", progress=False)
    except Exception as exc:
        print(f"Failed to download {ticker_symbol}: {exc}")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30"])

    if df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume", "returns", "sma_20", "sma_50", "volatility_30"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if "Date" in df.columns:
        df = df.reset_index(drop=True)
        df.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
    elif "date" in df.columns:
        df = df.reset_index(drop=True)
        df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"}, inplace=True)
    else:
        df = df.reset_index()
        df.rename(columns={"index": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    df["returns"] = df["close"].pct_change()
    df["sma_20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["sma_50"] = df["close"].rolling(window=50, min_periods=1).mean()
    df["volatility_30"] = df["returns"].rolling(window=30, min_periods=1).std()
    df[["sma_20", "sma_50", "volatility_30"]] = df[["sma_20", "sma_50", "volatility_30"]].ffill().fillna(0.0)
    df["returns"] = df["returns"].fillna(0.0)
    df["stock_name"] = stock_name
    return df

def main():
    parser = argparse.ArgumentParser(description="AuraTrade AI Data Pipeline")
    parser.add_argument("--tickers", nargs="+", required=True, help="List of stock tickers to download")
    parser.add_argument("--index", required=True, help="Benchmark index ticker (e.g. ^NSEI)")
    parser.add_argument("--years", type=int, default=5, help="Number of years of history to download")
    parser.add_argument("--out", required=True, help="Output directory for raw cached data")
    parser.add_argument(
        "--use-finbert",
        action="store_true",
        help="Use FinBERT for sentiment (downloads model from Hugging Face; slow first run)",
    )

    args = parser.parse_args()
    
    # Setup directories
    os.makedirs(args.out, exist_ok=True)
    
    # Define date range
    # yfinance treats end= as exclusive, so include the latest available market day.
    end_date = datetime.date.today() + datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=args.years * 365)
    
    print(f"Data Pipeline starting. Period: {start_date} to {end_date}")
    
    # 1. Download benchmark index data
    print(f"Downloading benchmark index returns for {args.index}...")
    try:
        df_index = yf.download(args.index, start=start_date, end=end_date)
        if isinstance(df_index.columns, pd.MultiIndex):
            df_index.columns = df_index.columns.get_level_values(0)
        df_index = df_index.reset_index()
        df_index.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
        df_index.sort_values("date", inplace=True)
        df_index.to_csv(os.path.join(args.out, "index_prices.csv"), index=False)
        print(f"Saved index prices: {len(df_index)} rows")
    except Exception as e:
        print(f"CRITICAL: Failed to download index {args.index}: {e}")
        return
        
    # 2. Download macro/exogenous variables
    print("Downloading macro/exogenous variables...")
    macro_df_list = []
    for macro_name, ticker in MACRO_TICKERS.items():
        try:
            print(f"  Downloading {macro_name} ({ticker})...")
            df_m = yf.download(ticker, start=start_date, end=end_date)
            if isinstance(df_m.columns, pd.MultiIndex):
                df_m.columns = df_m.columns.get_level_values(0)
            df_m = df_m.reset_index()
            # Select Close
            df_m = df_m[["Date", "Close"]].copy()
            df_m.rename(columns={"Date": "date", "Close": macro_name}, inplace=True)
            macro_df_list.append(df_m)
        except Exception as e:
            print(f"  Warning: Failed to download macro ticker {ticker}: {e}")
            
    if macro_df_list:
        df_macro = macro_df_list[0]
        for next_df in macro_df_list[1:]:
            df_macro = pd.merge(df_macro, next_df, on="date", how="outer")
        df_macro.sort_values("date", inplace=True)
        # Forward fill and backward fill macro data to handle non-trading alignments
        df_macro.ffill(inplace=True)
        df_macro.bfill(inplace=True)
        df_macro.to_csv(os.path.join(args.out, "macro_variables.csv"), index=False)
        print(f"Saved macro variables: {len(df_macro)} rows")
    else:
        print("Warning: No macro variables downloaded.")

    sentiment_analyzer = SentimentAnalyzer(use_transformer=args.use_finbert)
    print(f"Sentiment backend: {sentiment_analyzer.backend}")
    
    # 3. Download stock-specific data
    for ticker in args.tickers:
        print(f"\nProcessing Ticker: {ticker}")
        
        # A. Download prices
        try:
            df_stock = yf.download(ticker, start=start_date, end=end_date)
            if df_stock.empty:
                print(f"  Error: yfinance returned empty DataFrame for {ticker}")
                continue
                
            if isinstance(df_stock.columns, pd.MultiIndex):
                df_stock.columns = df_stock.columns.get_level_values(0)
            df_stock = df_stock.reset_index()
            df_stock.rename(columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}, inplace=True)
            df_stock.sort_values("date", inplace=True)
            
            # Save raw stock prices
            price_out = os.path.join(args.out, f"{ticker}_prices.csv")
            df_stock.to_csv(price_out, index=False)
            print(f"  Saved prices: {len(df_stock)} rows to {price_out}")
            
        except Exception as e:
            print(f"  Error downloading prices for {ticker}: {e}")
            continue

        # B. Download Corporate actions and Earnings calendar from yfinance
        try:
            print(f"  Fetching corporate events for {ticker}...")
            yf_ticker = yf.Ticker(ticker)
            
            # Get splits and dividends (yfinance may return only one of the two columns)
            actions = yf_ticker.actions
            if actions is not None and not actions.empty:
                actions = actions.reset_index()
                actions.rename(
                    columns={"Date": "date", "Stock Splits": "stock_splits", "Dividends": "dividends"},
                    inplace=True,
                )
                if "date" not in actions.columns:
                    actions.rename(columns={actions.columns[0]: "date"}, inplace=True)
                for col in ("stock_splits", "dividends"):
                    if col not in actions.columns:
                        actions[col] = 0.0
                actions = actions[["date", "stock_splits", "dividends"]].copy()
                actions["date"] = pd.to_datetime(actions["date"]).dt.tz_localize(None)
            else:
                actions = pd.DataFrame(columns=["date", "stock_splits", "dividends"])
                
            # Get earnings dates
            earnings_dates = []
            try:
                calendar = yf_ticker.calendar
                # For some tickers calendar could be list, dict or df depending on yfinance version
                if isinstance(calendar, dict) and "Earnings Date" in calendar:
                    earnings_dates = calendar["Earnings Date"]
                elif isinstance(calendar, pd.DataFrame) and "Earnings Date" in calendar.index:
                    earnings_dates = list(calendar.loc["Earnings Date"])
            except Exception:
                pass
                
            df_earnings = pd.DataFrame(earnings_dates, columns=["date"]) if earnings_dates else pd.DataFrame(columns=["date"])
            if not df_earnings.empty:
                df_earnings["date"] = pd.to_datetime(df_earnings["date"]).dt.tz_localize(None)
                df_earnings["is_earnings_day"] = 1.0
            else:
                df_earnings["is_earnings_day"] = pd.Series(dtype=float)
                
            # Merge events
            if not actions.empty or not df_earnings.empty:
                if actions.empty:
                    df_events = df_earnings
                elif df_earnings.empty:
                    df_events = actions
                else:
                    df_events = pd.merge(actions, df_earnings, on="date", how="outer")
                for col in ("stock_splits", "dividends", "is_earnings_day"):
                    if col not in df_events.columns:
                        df_events[col] = 0.0
                df_events.sort_values("date", inplace=True)
                events_out = os.path.join(args.out, f"{ticker}_events.csv")
                df_events.to_csv(events_out, index=False)
                print(f"  Saved corporate events to {events_out}")
            else:
                # Save empty schema
                pd.DataFrame(columns=["date", "stock_splits", "dividends", "is_earnings_day"]).to_csv(os.path.join(args.out, f"{ticker}_events.csv"), index=False)
                print("  No corporate events found. Saved empty events schema.")
                
        except Exception as e:
            print(f"  Warning: Failed to fetch corporate events for {ticker}: {e}")
            # Save empty schema as fallback
            pd.DataFrame(columns=["date", "stock_splits", "dividends", "is_earnings_day"]).to_csv(os.path.join(args.out, f"{ticker}_events.csv"), index=False)

        # C. Fetch News RSS headlines and score with FinBERT (lexicon fallback if unavailable)
        try:
            print(f"  Fetching news headlines for {ticker}...")
            daily_sentiment = sentiment_analyzer.score_ticker_news(ticker)
            sentiment_out = os.path.join(args.out, f"{ticker}_sentiment.csv")

            if not daily_sentiment.empty:
                daily_sentiment.to_csv(sentiment_out, index=False)
                print(f"    Saved {len(daily_sentiment)} daily sentiment rows to {sentiment_out}")
            else:
                pd.DataFrame(columns=["date", "sentiment_score"]).to_csv(sentiment_out, index=False)
                print("    No news headlines found. Saved empty sentiment schema.")
        except Exception as e:
            print(f"  Warning: news fetching failed for {ticker}: {e}")
            pd.DataFrame(columns=["date", "sentiment_score"]).to_csv(
                os.path.join(args.out, f"{ticker}_sentiment.csv"), index=False
            )

    print("\nData Pipeline Complete! All raw data saved to:", args.out)

if __name__ == "__main__":
    main()
