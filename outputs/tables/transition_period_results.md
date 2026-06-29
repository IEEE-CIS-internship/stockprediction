# Layer 3: Transition-Period Accuracy (+/- 5 Days) (Headline Metric)

| Horizon   | Model             |      RMSE |       MAE |      MAPE |
|:----------|:------------------|----------:|----------:|----------:|
| 1d        | Linear_Regression | 0.562426  | 0.42262   | 35703.9   |
| 1d        | Plain_LSTM        | 0.0220376 | 0.0137966 |   932.121 |
| 1d        | LSTM_Sentiment    | 0.0223285 | 0.0145656 |   671.627 |
| 1d        | Regime_MoE        | 0.0231986 | 0.0139138 |   990.233 |
| 3d        | Linear_Regression | 1.83171   | 1.19417   | 30653     |
| 3d        | Plain_LSTM        | 0.0443919 | 0.0305119 |   476.995 |
| 3d        | LSTM_Sentiment    | 0.0363271 | 0.0246027 |   427.676 |
| 3d        | Regime_MoE        | 0.0347409 | 0.0226069 |   319.676 |
| 7d        | Linear_Regression | 3.70001   | 2.42376   | 35275.2   |
| 7d        | Plain_LSTM        | 0.0635323 | 0.0492837 |   752.216 |
| 7d        | LSTM_Sentiment    | 0.0600621 | 0.0448811 |   578.467 |
| 7d        | Regime_MoE        | 0.0596624 | 0.0449708 |   465.822 |