# Model Card

## Sales model
- Algorithm: LightGBM regressor
- Target: log1p(Sales)
- Validation: July 2015 chronological holdout
- RMSLE: 0.0535
- MAE: 256.24
- RMSPE: 0.0573
- Training rows for final fit: 1,017,209

## Customer model
- Algorithm: LightGBM regressor
- Target: log1p(Customers)
- Training rows: 844,392 open-store rows
- Closed rows are forced to zero.

## LSTM benchmark
- Framework: PyTorch
- Architecture: 2 LSTM layers, hidden size 64
- Window: 30 days
- Scaling: MinMaxScaler(-1, 1)
- Stationarity: ADF p-value 6.44e-05; differencing not required
- Series: aggregate daily Sales

## Caveats
- Raw promotion comparisons are associative, not causal.
- Customers is not used as a sales predictor because it is unavailable in test.
- The 11 missing Open values in test are treated as closed.
