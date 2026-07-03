import argparse
import urllib.parse

import feedparser
import pandas as pd

# Financial sentiment lexicons for rule-based fallback
POSITIVE_WORDS = {
    "bullish", "growth", "profit", "gain", "rise", "surge", "higher", "positive",
    "beat", "strong", "outperform", "buy", "lead", "expand", "record", "recovery",
}
NEGATIVE_WORDS = {
    "bearish", "loss", "decline", "fall", "drop", "lower", "negative", "miss",
    "weak", "underperform", "sell", "lag", "shrink", "crash", "slump", "concern",
}

FINBERT_MODEL_ID = "ProsusAI/finbert"


class SentimentAnalyzer:
    def __init__(self, use_transformer: bool = False):
        self.use_transformer = use_transformer
        self.tokenizer = None
        self.model = None
        self.nlp = None

        if self.use_transformer:
            try:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

                print(f"Loading FinBERT model ({FINBERT_MODEL_ID}) from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL_ID)
                self.model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_ID)
                self.nlp = pipeline(
                    "sentiment-analysis",
                    model=self.model,
                    tokenizer=self.tokenizer,
                    device=-1,
                    truncation=True,
                )
                print("FinBERT model loaded successfully.")
            except Exception as exc:
                print(f"Failed to load FinBERT model: {exc}. Falling back to lexicon sentiment.")
                self.use_transformer = False

    @property
    def backend(self) -> str:
        return "finbert" if self.use_transformer and self.nlp else "lexicon"

    def get_google_news_headlines(self, query: str, num_results: int = 100) -> pd.DataFrame:
        """Fetch recent headlines from Google News RSS."""
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"

        headlines = []
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:num_results]:
                title = entry.title
                pub_date = entry.published
                clean_title = title.split(" - ")[0] if " - " in title else title
                parsed_date = pd.to_datetime(pub_date).strftime("%Y-%m-%d")
                headlines.append({"date": parsed_date, "headline": clean_title})
        except Exception as exc:
            print(f"Error fetching RSS news for '{query}': {exc}")

        return pd.DataFrame(headlines)

    def analyze_sentiment_fallback(self, text: str) -> float:
        words = text.lower().split()
        pos_count = sum(1 for word in words if word in POSITIVE_WORDS)
        neg_count = sum(1 for word in words if word in NEGATIVE_WORDS)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    @staticmethod
    def _finbert_label_to_score(label: str, score: float) -> float:
        normalized = label.lower()
        if normalized == "positive":
            return score
        if normalized == "negative":
            return -score
        return 0.0

    def analyze_sentiment(self, text: str) -> float:
        """Score headline sentiment in [-1, 1]. Uses FinBERT when enabled."""
        if not text:
            return 0.0

        if self.use_transformer and self.nlp:
            try:
                result = self.nlp(text[:512])[0]
                return self._finbert_label_to_score(result["label"], float(result["score"]))
            except Exception:
                return self.analyze_sentiment_fallback(text)

        return self.analyze_sentiment_fallback(text)

    def get_daily_sentiment(self, df_headlines: pd.DataFrame) -> pd.DataFrame:
        if df_headlines.empty:
            return pd.DataFrame(columns=["date", "sentiment_score"])

        scored = df_headlines.copy()
        scored["sentiment_score"] = scored["headline"].apply(self.analyze_sentiment)
        daily_sentiment = scored.groupby("date")["sentiment_score"].mean().reset_index()
        return daily_sentiment

    def score_ticker_news(self, ticker: str, query_suffix: str = "stock India") -> pd.DataFrame:
        clean_name = ticker.split(".")[0]
        query = f"{clean_name} {query_suffix}".strip()
        df_headlines = self.get_google_news_headlines(query)
        return self.get_daily_sentiment(df_headlines)


def fetch_and_process_sentiment(ticker_name: str, use_transformer: bool = False) -> pd.DataFrame | None:
    analyzer = SentimentAnalyzer(use_transformer=use_transformer)
    print(f"Fetching news for {ticker_name} using {analyzer.backend} sentiment...")
    df_headlines = analyzer.get_google_news_headlines(f"{ticker_name} stock India")
    df_sentiment = analyzer.get_daily_sentiment(df_headlines)

    if df_sentiment.empty:
        print(f"No news headlines found for {ticker_name}.")
        return None

    print(f"Processed {len(df_sentiment)} days of news sentiment for {ticker_name}.")
    return df_sentiment


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score Google News RSS sentiment for a ticker")
    parser.add_argument("ticker", help="Ticker symbol, e.g. ICICIBANK.NS")
    parser.add_argument("--use-finbert", action="store_true", help="Use FinBERT instead of lexicon sentiment")
    cli_args = parser.parse_args()
    result = fetch_and_process_sentiment(cli_args.ticker, use_transformer=cli_args.use_finbert)
    if result is not None:
        print(result.head())
