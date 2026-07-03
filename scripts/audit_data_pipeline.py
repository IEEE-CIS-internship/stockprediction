"""Audit raw pipeline outputs and live yFinance availability per ticker."""
import datetime
import os
import sys

import pandas as pd
import yfinance as yf

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
FEATURES_DIR = os.path.join(BASE_DIR, "data", "processed", "features")

STOCK_UNIVERSE = [
    "ICICIBANK.NS",
    "HCLTECH.NS",
    "INDIGO.NS",
    "MARUTI.NS",
    "TRENT.NS",
    "HINDALCO.NS",
    "ONGC.NS",
    "ADANIENT.NS",
]

REQUIRED_PRICE_COLS = ["date", "open", "high", "low", "close", "volume"]
MIN_PRICE_ROWS = 200


def audit_cached_ticker(ticker: str) -> dict:
    result = {"ticker": ticker, "cached_ok": True, "issues": []}

    for suffix in ("_prices.csv", "_events.csv", "_sentiment.csv"):
        path = os.path.join(RAW_DIR, f"{ticker}{suffix}")
        if not os.path.exists(path):
            result["cached_ok"] = False
            result["issues"].append(f"missing {suffix}")
            continue

        df = pd.read_csv(path)
        if suffix == "_prices.csv":
            missing_cols = [c for c in REQUIRED_PRICE_COLS if c not in df.columns]
            if missing_cols:
                result["cached_ok"] = False
                result["issues"].append(f"prices missing columns: {missing_cols}")
            if len(df) < MIN_PRICE_ROWS:
                result["cached_ok"] = False
                result["issues"].append(f"prices only {len(df)} rows (need >={MIN_PRICE_ROWS})")
            null_ohlc = df[["open", "high", "low", "close"]].isna().sum().sum()
            if null_ohlc:
                result["cached_ok"] = False
                result["issues"].append(f"prices have {null_ohlc} null OHLC values")
            df["date"] = pd.to_datetime(df["date"])
            result["price_rows"] = len(df)
            result["price_start"] = str(df["date"].min().date())
            result["price_end"] = str(df["date"].max().date())
            result["latest_close"] = float(df["close"].iloc[-1])
        elif suffix == "_events.csv" and df.empty:
            result["issues"].append("events file empty (non-fatal)")
        elif suffix == "_sentiment.csv" and df.empty:
            result["issues"].append("sentiment file empty (non-fatal)")

    features_path = os.path.join(FEATURES_DIR, f"{ticker}_features.csv")
    if os.path.exists(features_path):
        fdf = pd.read_csv(features_path)
        result["feature_rows"] = len(fdf)
        result["features_ok"] = len(fdf) >= MIN_PRICE_ROWS
        if not result["features_ok"]:
            result["cached_ok"] = False
            result["issues"].append(f"features only {len(fdf)} rows")
    else:
        result["features_ok"] = False
        result["cached_ok"] = False
        result["issues"].append("missing processed features file")

    return result


def audit_live_yfinance(ticker: str) -> dict:
    end_date = datetime.date.today() + datetime.timedelta(days=1)
    start_date = end_date - datetime.timedelta(days=5 * 365)
    result = {"ticker": ticker, "live_ok": True, "live_issues": []}

    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            result["live_ok"] = False
            result["live_issues"].append("yfinance returned empty DataFrame")
            return result
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        result["live_rows"] = len(df)
        if len(df) < MIN_PRICE_ROWS:
            result["live_ok"] = False
            result["live_issues"].append(f"only {len(df)} live rows")
        info = yf.Ticker(ticker).info
        short_name = info.get("shortName") or info.get("longName")
        if not short_name:
            result["live_issues"].append("ticker info has no name (may still be valid)")
        else:
            result["live_name"] = short_name
    except Exception as exc:
        result["live_ok"] = False
        result["live_issues"].append(str(exc))

    return result


def main() -> int:
    print("=== AuraTrade data pipeline audit ===\n")

    for path, label in [
        (os.path.join(RAW_DIR, "index_prices.csv"), "index"),
        (os.path.join(RAW_DIR, "macro_variables.csv"), "macro"),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"{label}: OK ({len(df)} rows)")
        else:
            print(f"{label}: MISSING")
            return 1

    all_cached_ok = True
    all_live_ok = True

    print("\nPer-stock summary:")
    print(f"{'Ticker':<16} {'Cached':<8} {'Live':<8} {'Rows':<6} {'Date range':<24} Issues")
    print("-" * 90)

    for ticker in STOCK_UNIVERSE:
        cached = audit_cached_ticker(ticker)
        live = audit_live_yfinance(ticker)
        all_cached_ok &= cached["cached_ok"]
        all_live_ok &= live["live_ok"]

        date_range = ""
        if "price_start" in cached:
            date_range = f"{cached['price_start']} -> {cached['price_end']}"

        issues = "; ".join(cached.get("issues", []) + live.get("live_issues", [])) or "-"
        rows = cached.get("price_rows", live.get("live_rows", "?"))

        print(
            f"{ticker:<16} "
            f"{'OK' if cached['cached_ok'] else 'FAIL':<8} "
            f"{'OK' if live['live_ok'] else 'FAIL':<8} "
            f"{rows:<6} "
            f"{date_range:<24} "
            f"{issues}"
        )

    print("\n=== Verdict ===")
    if all_cached_ok and all_live_ok:
        print("PASS: Cached pipeline data and live yFinance fetches look good for all 8 stocks.")
        return 0

    if not all_cached_ok:
        print("FAIL: One or more cached pipeline files are missing or low quality.")
    if not all_live_ok:
        print("FAIL: One or more live yFinance downloads failed.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
