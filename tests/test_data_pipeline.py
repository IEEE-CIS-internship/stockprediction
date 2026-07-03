import os
import sys
import unittest
from unittest.mock import patch

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
import data_pipeline


class DataPipelineTests(unittest.TestCase):
    @patch("data_pipeline.yf.download")
    def test_download_stock_data_returns_expected_columns(self, mock_download):
        mock_download.return_value = pd.DataFrame(
            {
                "Open": [100.0, 101.0],
                "High": [101.0, 102.0],
                "Low": [99.5, 100.5],
                "Close": [100.5, 101.5],
                "Volume": [1000, 1200],
            },
            index=[pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")],
        )

        df = data_pipeline.download_stock_data("Test Stock", "TEST.NS")

        self.assertIn("date", df.columns)
        self.assertIn("close", df.columns)
        self.assertIn("returns", df.columns)
        self.assertIn("sma_20", df.columns)
        self.assertIn("sma_50", df.columns)
        self.assertIn("volatility_30", df.columns)
        self.assertGreaterEqual(len(df), 2)

    def test_corporate_actions_with_dividends_only(self):
        actions = pd.DataFrame(
            {"Dividends": [2.0, 2.5]},
            index=pd.to_datetime(["2024-01-02", "2024-06-01"]),
        )
        actions = actions.reset_index().rename(columns={"index": "Date"})
        actions.rename(
            columns={"Date": "date", "Stock Splits": "stock_splits", "Dividends": "dividends"},
            inplace=True,
        )
        for col in ("stock_splits", "dividends"):
            if col not in actions.columns:
                actions[col] = 0.0
        actions = actions[["date", "stock_splits", "dividends"]]

        self.assertIn("stock_splits", actions.columns)
        self.assertIn("dividends", actions.columns)
        self.assertEqual(actions["stock_splits"].tolist(), [0.0, 0.0])
        self.assertEqual(actions["dividends"].tolist(), [2.0, 2.5])


if __name__ == "__main__":
    unittest.main()
