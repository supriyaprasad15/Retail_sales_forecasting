# Rossmann Sales Forecasting — End-to-End ML + Deep Learning

This project implements the full workflow requested in the supplied project brief: EDA, preprocessing and feature creation, sklearn-compatible modelling, loss-function justification, feature importance, prediction intervals, model serialization, a two-layer LSTM time-series benchmark, MLflow experiment tracking, DVC workflow, and a Flask prediction dashboard.

## 1. Dataset

The supplied files are:
- `data/train.csv` — labelled historical store-day data.
- `data/test.csv` — future store-day rows to forecast.
- `data/store.csv` — store metadata.
- `data/sample_submission.csv` — required submission format.

The brief defines Sales as the daily turnover target and identifies Store, Customers, Open, promotions, holidays, assortment, competitor distance, and Promo2 as important fields. See the project PDF, pages 1–2.

### Dataset audit performed
- Train: 1,017,209 rows, 1115 stores; dates 2013-01-01 to 2015-07-31.
- Test: 41,088 rows, 856 stores; dates 2015-08-01 to 2015-09-17.
- Store metadata: 1,115 stores.
- Train Sales zeros: 16.99%.
- Test Open has 11 missing values; these are conservatively treated as closed (`Open=0`) in the submission pipeline. They belong to Store 622.

## 2. Key EDA findings

1. Promo rate is 38.15% in train and 39.58% in test, so promotion prevalence is broadly similar.
2. Sales and Customers correlation for open stores is 0.824.
3. Mean open-store sales are 5,929 without Promo and 8,228 with Promo.
4. Mean open-store customers are 697 without Promo and 844 with Promo.
5. December is the strongest month in this dataset by average open-store sales (8,609), supporting strong seasonality.
6. Only 10 stores are open on every observed weekday; their mean weekend sales are about 11,627, versus 2,953 for other stores.
7. `store.csv` has 3 missing CompetitionDistance values (Stores 291, 622, 879). The supplied store table is static, so it does not contain a time-varying distance history with which to prove a later competitor opening/reopening.

## 3. Modelling strategy

### Production ML model
The final model is LightGBM wrapped in a sklearn `Pipeline`. The target is `log1p(Sales)` to reduce the effect of the heavy right tail; predictions are transformed back with `expm1`.

Feature engineering includes:
- year, month, day, week of year, day of year, quarter
- weekday/weekend
- beginning/middle/end of month
- cyclical month and weekday features
- competition age
- Promo2 age and active month
- days since/to the nearest observed state holiday
- Store × Promo interaction
- store metadata from `store.csv`

`Customers` is deliberately **not** used as a sales feature because Customers is present in train but absent from test; using it would create target-time information mismatch/leakage.

### Validation
A chronological validation split uses July 2015 as the holdout, matching the forecasting nature of the problem rather than randomly shuffling time.

Validation results from the trained model:
- RMSLE: 0.0535
- MAE: 256.24
- RMSPE: 0.0573

RMSLE is the main loss because sales are non-negative and highly skewed, and the log scale treats proportional errors more evenly. RMSPE is also reported because it is closely aligned with the traditional Rossmann forecasting evaluation.

### Customers model
A second LightGBM pipeline is trained on open-store rows with `log1p(Customers)` as target. Closed stores are forced to zero.

## 4. Deep Learning

The brief asks for a two-layer LSTM workflow. The implementation:
1. aggregates historical sales into a daily time series,
2. checks stationarity with ADF,
3. checks ACF/PACF,
4. creates 30-day sliding windows,
5. scales to (-1, 1),
6. trains a two-layer PyTorch LSTM.

For this dataset, the ADF p-value on aggregate daily sales is 6.44e-05, below 0.05, so differencing is not required. A log transform is used to stabilize variance.

The LSTM is included as the required deep-learning benchmark. The production store-level forecast remains the LightGBM model because it naturally incorporates store metadata, promotions, holidays and locality for every store/test row.

## 5. Confidence intervals

The project includes a practical prediction-interval approach for the production model: use validation residuals on the original sales scale, grouped by store type and promotion state when enough validation observations exist, and take empirical quantiles around each point prediction. This is a conformal-style residual approach rather than a parametric Gaussian assumption.

A production implementation can be enabled in `src/intervals.py` (included as a reference implementation).

## 6. Dashboard

Run:
```bash
pip install -r requirements.txt
python app.py
```
Then open `http://127.0.0.1:5000`.

The Flask dashboard supports:
- Store ID + date input
- Open/Promo/holiday inputs
- CSV upload
- predicted sales
- predicted customers
- prediction chart
- downloadable CSV

## 7. MLflow

Install MLflow and run:
```bash
python -m src.mlflow_tracking
mlflow ui
```
This creates multiple LightGBM runs with different `num_leaves`, logs metrics and parameters, and stores model artifacts. Capture the MLflow UI screenshot for the final submission.

## 8. DVC

After placing the supplied data into `data/`:
```bash
git init
dvc init
dvc add data/train.csv data/test.csv data/store.csv data/sample_submission.csv
git add .
git commit -m "Track Rossmann data with DVC"
```
For a remote store, configure the remote and run `dvc push`. Capture the DVC history/remote evidence requested by the brief.

## 9. Reproduce

```bash
python -m src.train_ml
python -m src.train_lstm
python -m src.mlflow_tracking
python app.py
```

## 10. Deliverables

- `notebooks/rossmann_end_to_end.ipynb`
- `models/sales_model.pkl`
- `models/customers_model.pkl`
- `models/lstm_sales_model.pt`
- `models/lstm_scaler.pkl`
- `outputs/submission.csv`
- `outputs/customer_sales_predictions.csv`
- EDA and LSTM plots under `outputs/`
- `report.pdf`
- `interim_slides.pptx`

## Project-brief alignment

The supplied PDF explicitly asks for EDA, prediction, ML and deep learning approaches, web serving, logging, serialization, MLflow, DVC evidence, a final PDF/blog, GitHub code and a deployed application. The implementation is organized to cover those requirements.
