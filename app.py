from pathlib import Path
import io
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, send_file, flash

ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)
app = Flask(__name__)
app.secret_key = "rossmann-demo-secret"

sales_model = joblib.load(ROOT / "models" / "sales_model.pkl")
customers_model = joblib.load(ROOT / "models" / "customers_model.pkl")


def prepare_input(df, store_id=None):
    df = df.copy()
    rename = {"Store_id": "Store", "StoreID": "Store",
              "IsHoliday": "StateHoliday", "IsPromo": "Promo"}
    df.rename(columns={k: v for k, v in rename.items() if k in df.columns}, inplace=True)
    if "Store" not in df.columns:
        if store_id is None:
            raise ValueError("Store or Store_id is required.")
        df["Store"] = int(store_id)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    if df["Date"].isna().any():
        raise ValueError("Invalid Date value found.")
    if "DayOfWeek" not in df.columns:
        df["DayOfWeek"] = df["Date"].dt.dayofweek + 1
    if "Open" not in df.columns:
        df["Open"] = 1
    if "Promo" not in df.columns:
        df["Promo"] = 0
    if "StateHoliday" not in df.columns:
        df["StateHoliday"] = "0"
    if "SchoolHoliday" not in df.columns:
        df["SchoolHoliday"] = 0
    df["Open"] = pd.to_numeric(df["Open"], errors="coerce").fillna(0).astype(int)
    df["Promo"] = pd.to_numeric(df["Promo"], errors="coerce").fillna(0).astype(int)
    df["SchoolHoliday"] = pd.to_numeric(df["SchoolHoliday"], errors="coerce").fillna(0).astype(int)
    return df


def predict(df):
    df = prepare_input(df)
    sales = np.maximum(0, np.expm1(sales_model.predict(df)))
    customers = np.maximum(0, np.expm1(customers_model.predict(df)))
    sales = np.where(df["Open"].values == 1, sales, 0)
    customers = np.where(df["Open"].values == 1, customers, 0)
    result = df[["Store", "Date"]].copy()
    result["PredictedSales"] = np.rint(sales).astype(int)
    result["PredictedCustomers"] = np.rint(customers).astype(int)
    return result


@app.route("/download")
def download():
    path = OUTPUTS / "latest_predictions.csv"
    if not path.exists():
        return "No prediction has been generated yet.", 404
    return send_file(path, mimetype="text/csv", as_attachment=True,
                     download_name="rossmann_predictions.csv")


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        try:
            uploaded = request.files.get("file")
            if uploaded and uploaded.filename:
                df = pd.read_csv(uploaded)
            else:
                df = pd.DataFrame([{
                    "Store": request.form.get("store_id", type=int),
                    "Date": request.form["date"],
                    "Open": request.form.get("open", type=int),
                    "Promo": request.form.get("promo", type=int),
                    "StateHoliday": request.form.get("state_holiday", "0"),
                    "SchoolHoliday": request.form.get("school_holiday", type=int) or 0
                }])
            result = predict(df)
            result.to_csv(OUTPUTS / "latest_predictions.csv", index=False)
            result["Date"] = result["Date"].dt.strftime("%Y-%m-%d")
        except Exception as exc:
            flash(str(exc))
    return render_template("index.html",
                           result=result.to_dict("records") if result is not None else None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
