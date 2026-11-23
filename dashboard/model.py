"""The saved Stacking 1 model: loads the exported files, builds the inputs for a day, forecasts and explains."""
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

ART = Path(__file__).resolve().parent / "artifacts"
SEED = 42
PERIODS = ["Teaching", "Mid-Semester Break", "Semester Break"]


# ---------- Exported files ----------

@lru_cache(maxsize=None)
def meta():
    # Column order, scaling values, lag fill values, Ridge weights and dates
    return json.loads((ART / "metadata.json").read_text())


@lru_cache(maxsize=None)
def base_models():
    # The three base models fitted inside Stacking 1
    xgb = XGBRegressor()
    xgb.load_model(ART / "xgb.json")
    return joblib.load(ART / "rf.joblib"), xgb, joblib.load(ART / "svr.joblib")


@lru_cache(maxsize=None)
def csv(name):
    # One of the exported tables, indexed by date
    frame = pd.read_csv(ART / f"{name}.csv")
    frame["Timestamp"] = pd.to_datetime(frame["Timestamp"])
    return frame.set_index("Timestamp")


@lru_cache(maxsize=None)
def table(name):
    # Results, tuning comparison or global importance table from the notebook
    header = [0, 1] if name == "global_importance" else 0
    index = [0, 1] if name == "tuning_comparison" else 0
    return pd.read_csv(ART / f"{name}.csv", header=header, index_col=index)


def history():
    # Every day of the dataset: calendar, price, meals sold and lag values
    return csv("history")


def test_start():
    # First day of the test period, where the daily log starts
    return pd.Timestamp(meta()["test_dates"][0])


def last_data_day():
    return history().index[-1]


# ---------- Calendar ----------

def next_operating_day(day, closed=()):
    # The next day the canteen opens: Monday to Saturday, skipping any days marked as closed
    closed = {pd.Timestamp(d) for d in closed}
    day = pd.Timestamp(day) + pd.Timedelta(days=1)
    while day.dayofweek == 6 or day in closed:
        day += pd.Timedelta(days=1)
    return day


def period_of(day):
    # Academic period from the dataset's calendar, or None for days after the data ends
    day = pd.Timestamp(day)
    return history().loc[day, "Academic_Period"] if day in history().index else None


def price_of(day):
    # Recorded average meal price of a day, or the training average for days after the data ends
    day = pd.Timestamp(day)
    if day in history().index:
        return float(history().loc[day, "Price_per_meal"])
    start, end = meta()["train_dates"]
    return float(history().loc[start:end, "Price_per_meal"].mean())


# ---------- Inputs and forecast ----------

def conditions(day, sales, period=None, price=None):
    # Everything the model needs for one day: calendar, price and the sales of the previous operating days.
    # sales: meals sold per earlier day (a Series indexed by date)
    day = pd.Timestamp(day)
    m = meta()
    past = sales[sales.index < day].sort_index()
    lags = {"Quantity_last_day": past.iloc[-1:], "Quantity_last_two_days": past.iloc[-2:],
            "Quantity_last_week": past.iloc[-m["week_days"]:]}
    need = {"Quantity_last_day": 1, "Quantity_last_two_days": 2, "Quantity_last_week": m["week_days"]}
    cond = {"Day_of_Week": day.day_name(), "Month": day.month_name(),
            "Weekend_Status": "Weekend" if day.dayofweek == 5 else "Weekday",
            "Academic_Period": period or period_of(day) or PERIODS[0],
            "Price_per_meal": price_of(day) if price is None else float(price)}
    for c, vals in lags.items():
        cond[c] = float(vals.sum()) if len(vals) == need[c] else float(m["lag_fill"][c])
    return cond


def inputs(cond):
    # One-hot encoded, min-max scaled input row in the notebook's column order
    m = meta()
    raw = []
    for col in m["columns"]:
        if col in cond:
            raw.append(float(cond[col]))
        else:
            cat = next(c for c in m["cat_cols"] if col.startswith(c + "_"))
            raw.append(1.0 if col == f"{cat}_{cond[cat]}" else 0.0)
    rng = np.array(m["scaler_range"])
    rng[rng == 0] = 1.0
    return ((np.array(raw) - np.array(m["scaler_min"])) / rng).reshape(1, -1)


def predict(X):
    # Stacking 1: Ridge weights times the RF, XGBoost and SVR forecasts, plus the intercept
    m = meta()
    rf, xgb, svr = base_models()
    X = np.atleast_2d(np.asarray(X, dtype=float))
    base = np.column_stack([rf.predict(X), xgb.predict(X), svr.predict(X) * m["svr_k"]])
    return base @ np.array(m["ridge_coef"]) + m["ridge_intercept"]


def forecast(cond):
    return float(predict(inputs(cond))[0])


def week_ahead(day, sales, days=6, period=None, closed=()):
    # Forecasts for the next operating days; later days use earlier forecasts in place of unknown sales
    sales = sales.copy()
    out = []
    for _ in range(days):
        fc = forecast(conditions(day, sales, period=period if period_of(day) is None else None))
        out.append({"Date": day, "Forecast": fc, "Period": period_of(day) or period or PERIODS[0]})
        sales.loc[day] = fc
        day = next_operating_day(day, closed)
    return pd.DataFrame(out)


# ---------- Explanations ----------

def variable_of(col):
    # Original variable a model column belongs to (one-hot columns map to their categorical variable)
    return next((c for c in meta()["cat_cols"] if col.startswith(c + "_")), col)


def explain(cond):
    # SHAP base value (a typical day) and the contribution of each original variable, in meals
    import shap
    sv = shap.PermutationExplainer(predict, np.load(ART / "shap_background.npy"), seed=SEED)(inputs(cond))
    contrib = pd.Series(sv.values[0], index=[variable_of(c) for c in meta()["columns"]]).groupby(level=0).sum()
    return float(sv.base_values[0]), contrib


def lime_table(cond, num_features=8):
    # LIME conditions and effects for one day, using the training days as reference data
    from lime.lime_tabular import LimeTabularExplainer
    start, end = meta()["train_dates"]
    train = csv("model_inputs").loc[start:end, meta()["columns"]].to_numpy()
    exp = LimeTabularExplainer(train, feature_names=meta()["columns"], mode="regression", random_state=SEED)
    result = exp.explain_instance(inputs(cond).ravel(), predict, num_features=num_features)
    return pd.DataFrame(result.as_list(), columns=["LIME condition (scaled inputs)", "Effect (meals)"])
