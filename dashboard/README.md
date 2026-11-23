# Canteen Meal Forecast Dashboard

A Streamlit dashboard for canteen staff. Each day it shows how many meals to prepare, the expected sales and a likely range, and the main reasons. At the end of the day staff record the meals sold (and, if they want, the meals prepared); the next day's forecast then uses those sales.

| Page | For | What it does |
|---|---|---|
| Today | Staff | Plan for the day, reason, special-day adjustment, week ahead, printable prep sheet, end-of-day record, closed days |
| Records | Manager | Recorded days, leftovers and shortfalls, warnings, weekly or monthly totals, notes and corrections, CSV download |
| What-if | Manager | How the forecast changes with a different academic period, price or recent sales |
| Performance | Examiner | Accuracy, training time, explanations, about and limitations |

The forecasts come from the stacking ensemble (Random Forest, XGBoost and SVR) trained in `canteen_methodology_improved.ipynb`. The dashboard loads the saved model from `artifacts/` and never retrains it. Recorded days are kept in `data/daily_log.csv` and closed days (for example public holidays) in `data/closed_days.csv`.

## Running on Windows

Double-click `run_dashboard.bat` in the repository folder. The first run installs the libraries (a few minutes; needs Python 3.13). The dashboard then opens in the browser. Close the window to stop it.

## Running manually

From the repository folder:

```bash
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py
```

## Online demo (optional)

The dashboard can be hosted free on Streamlit Community Cloud from the public GitHub repository: create an app, choose this repository, set the main file to `dashboard/app.py` and choose Python 3.13 in the advanced settings. Recorded days are lost whenever the hosted app restarts, so use it for demonstrations only.

## Updating the model

Rerun `canteen_methodology_improved.ipynb`; its last section rewrites `artifacts/`. Keep the library versions in `requirements.txt`, because the saved models only load reliably with the same scikit-learn version.

## Tests

```bash
pip install pytest
pytest dashboard/tests
```
