import os
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import urllib.parse

# Financial sentiment lexicons for rule-based fallback
POSITIVE_WORDS = {"bullish", "growth", "profit", "gain", "rise", "surge", "higher", "positive", "beat", "strong", "outperform", "buy", "lead", "expand", "record"}
NEGATIVE_WORDS = {"bearish", "loss", "decline", "fall", "drop", "lower", "negative", "miss", "weak", "underperform", "sell", "lag", "shrink", "crash", "slump"}

class SentimentAnalyzer:
    def __init__(self, use_transformer=True):
        self.use_transformer = use_transformer
        self.tokenizer = None
        self.model = None
        self.nlp = None
        
        if self.use_transformer:
            try:
                from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
                print("Loading FinBERT model from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
                self.model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
                self.nlp = pipeline("sentiment-analysis", model=self.model, tokenizer=self.tokenizer)
                print("FinBERT model loaded successfully.")
            except Exception as e:
                print(f"Failed to load FinBERT model: {e}. Falling back to Rule-Based Lexicon Sentiment.")
                self.use_transformer = False

    def get_google_news_headlines(self, query, num_results=100):
        """
        Scrapes Google News RSS feed for a specific query to extract headlines and dates.
        """
        encoded_query = urllib.parse.quote(query)
        url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en-IN&gl=IN&ceid=IN:en"
        
        headlines = []
        try:
            response = requests.get(url, timeout=10)
            soup = BeautifulSoup(response.content, features="xml")
            items = soup.find_all("item")
            
            for item in items[:num_results]:
                title = item.title.text
                pub_date = item.pubDate.text
                # Remove source name from title (typically separated by " - ")
                clean_title = title.split(" - ")[0] if " - " in title else title
                # Parse date to YYYY-MM-DD
                parsed_date = pd.to_datetime(pub_date).strftime("%Y-%m-%d")
                
                headlines.append({
                    "date": parsed_date,
                    "headline": clean_title
                })
        except Exception as e:
            print(f"Error fetching RSS news for {query}: {e}")
            
        return pd.DataFrame(headlines)

    def analyze_sentiment_fallback(self, text):
        """
        Lightweight lexicon-based fallback to score sentiment between -1 (bearish) and 1 (bullish).
        """
        words = text.lower().split()
        pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
        neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)
        total = pos_count + neg_count
        if total == 0:
            return 0.0
        return (pos_count - neg_count) / total

    def analyze_sentiment(self, text):
        """
        Scores sentiment. Returns a single scalar score in range [-1, 1].
        """
        if self.use_transformer and self.nlp:
            try:
                result = self.nlp(text)[0]
                label = result["label"]
                score = result["score"]
                # Map FinBERT labels: positive -> 1, negative -> -1, neutral -> 0
                if label == "positive":
                    return score
                elif label == "negative":
                    return -score
                else:
                    return 0.0
            except Exception as e:
                # Fallback to rule-based on model error
                return self.analyze_sentiment_fallback(text)
        else:
            return self.analyze_sentiment_fallback(text)

    def get_daily_sentiment(self, df_headlines):
        """
        Scores a list of headlines and averages sentiment per day.
        """
        if df_headlines.empty:
            return pd.DataFrame(columns=["date", "sentiment_score"])
        
        df_headlines["score"] = df_headlines["headline"].apply(self.analyze_sentiment)
        daily_sentiment = df_headlines.groupby("date")["score"].mean().reset_index()
        daily_sentiment.rename(columns={"score": "sentiment_score"}, inplace=True)
        return daily_sentiment

def fetch_and_process_sentiment(ticker_name, start_date="2020-01-01"):
    analyzer = SentimentAnalyzer(use_transformer=False) # Fallback model for fast initialization
    print(f"Fetching news for {ticker_name}...")
    df_headlines = analyzer.get_google_news_headlines(f"{ticker_name} stock India")
    
    if df_headlines.empty:
        print(f"No news headlines found for {ticker_name}.")
        return None
        
    df_sentiment = analyzer.get_daily_sentiment(df_headlines)
    print(f"Processed {len(df_sentiment)} days of news sentiment for {ticker_name}.")
    return df_sentiment

if __name__ == "__main__":
    # Test sentiment pipeline
    res = fetch_and_process_sentiment("ICICI Bank")
    if res is not None:
        print(res.head())
