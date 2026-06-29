# Layer 1: Overall Accuracy Scorecard (Residual Returns)

| Horizon   | Model             |      RMSE |       MAE |       MAPE |
|:----------|:------------------|----------:|----------:|-----------:|
| 1d        | Linear_Regression | 0.893918  | 0.593133  |  48966.5   |
| 1d        | Plain_LSTM        | 0.0221851 | 0.0150705 |    870.355 |
| 1d        | LSTM_Sentiment    | 0.0242096 | 0.0173016 |   1002.58  |
| 1d        | Regime_MoE        | 0.0254536 | 0.0166533 |    994.866 |
| 3d        | Linear_Regression | 1.64625   | 1.1174    | 151644     |
| 3d        | Plain_LSTM        | 0.0428283 | 0.031258  |  11061.4   |
| 3d        | LSTM_Sentiment    | 0.0375923 | 0.0269512 |   1216.68  |
| 3d        | Regime_MoE        | 0.0357967 | 0.0247421 |  23258.7   |
| 7d        | Linear_Regression | 4.08693   | 2.79618   |  48437.8   |
| 7d        | Plain_LSTM        | 0.0677415 | 0.0525746 |   1016.15  |
| 7d        | LSTM_Sentiment    | 0.0688938 | 0.053059  |   1104.18  |
| 7d        | Regime_MoE        | 0.0638127 | 0.0491982 |    493.376 |