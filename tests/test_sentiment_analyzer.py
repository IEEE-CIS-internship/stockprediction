import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from sentiment_analyzer import SentimentAnalyzer


class SentimentAnalyzerTests(unittest.TestCase):
    def test_finbert_label_mapping(self):
        self.assertEqual(SentimentAnalyzer._finbert_label_to_score("positive", 0.9), 0.9)
        self.assertEqual(SentimentAnalyzer._finbert_label_to_score("Negative", 0.8), -0.8)
        self.assertEqual(SentimentAnalyzer._finbert_label_to_score("neutral", 0.5), 0.0)

    def test_lexicon_fallback_scores_headline(self):
        analyzer = SentimentAnalyzer(use_transformer=False)
        bullish = analyzer.analyze_sentiment("Company reports strong profit growth and bullish outlook")
        bearish = analyzer.analyze_sentiment("Stock faces bearish decline after weak earnings miss")
        self.assertGreater(bullish, 0.0)
        self.assertLess(bearish, 0.0)

    @patch("sentiment_analyzer.SentimentAnalyzer.__init__", return_value=None)
    def test_analyze_sentiment_uses_finbert_when_loaded(self, _mock_init):
        analyzer = SentimentAnalyzer.__new__(SentimentAnalyzer)
        analyzer.use_transformer = True
        analyzer.nlp = MagicMock(return_value=[{"label": "positive", "score": 0.91}])
        score = SentimentAnalyzer.analyze_sentiment(analyzer, "Margins expand on record profit")
        self.assertAlmostEqual(score, 0.91)

    def test_daily_sentiment_averages_by_date(self):
        analyzer = SentimentAnalyzer(use_transformer=False)
        headlines = pd.DataFrame(
            {
                "date": ["2024-01-02", "2024-01-02", "2024-01-03"],
                "headline": [
                    "bullish growth profit",
                    "bearish loss decline",
                    "neutral update",
                ],
            }
        )
        daily = analyzer.get_daily_sentiment(headlines)
        self.assertEqual(len(daily), 2)
        self.assertIn("sentiment_score", daily.columns)


if __name__ == "__main__":
    unittest.main()
