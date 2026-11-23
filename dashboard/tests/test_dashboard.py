"""Checks that the dashboard reproduces the notebook and that the daily loop, advice and pages work."""
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))


@pytest.fixture(autouse=True)
def fresh_log(tmp_path, monkeypatch):
    # Every test starts with an empty daily log in a temporary folder
    monkeypatch.setenv("DASHBOARD_LOG", str(tmp_path / "log.csv"))
    monkeypatch.setenv("DASHBOARD_CLOSED", str(tmp_path / "closed.csv"))
    import store
    monkeypatch.setattr(store, "LOG", tmp_path / "log.csv")
    monkeypatch.setattr(store, "CLOSED", tmp_path / "closed.csv")
    yield


import insights  # noqa: E402
import model  # noqa: E402
import store  # noqa: E402


def record_recorded_days(k):
    # Record the first k test days with their recorded sales, as staff would; returns the forecasts shown
    shown = []
    for _ in range(k):
        day = store.current_day()
        fc = model.forecast(model.conditions(day, store.sales()))
        shown.append(fc)
        store.save(day, int(model.history().loc[day, model.meta()["target"]]), None, fc, insights.plan(fc))
    return shown


def test_inputs_match_notebook():
    # Inputs built from sales equal the notebook's scaled inputs, including the first day (missing lags) and a Monday
    mi, hist = model.csv("model_inputs"), model.history()
    sales = hist[model.meta()["target"]].astype(float)
    for d in [hist.index[0], hist.index[3], hist[hist.Day_of_Week == "Monday"].index[5], model.test_start(), hist.index[-1]]:
        built = model.inputs(model.conditions(d, sales)).ravel()
        assert np.allclose(built, mi.loc[d, model.meta()["columns"]].to_numpy(float), atol=1e-9), d


def test_parity_all_test_days():
    # Forecasts for every test day equal the notebook's
    hist = model.history()
    sales = hist[model.meta()["target"]].astype(float)
    tp = model.csv("test_predictions")
    fc = np.array([model.forecast(model.conditions(d, sales)) for d in tp.index])
    assert np.abs(fc - tp["Stacking 1"].to_numpy()).max() < 0.5


def test_daily_loop_reproduces_notebook():
    # Recording the recorded sales day by day gives exactly the notebook's forecasts for the next days
    shown = record_recorded_days(8)
    assert np.abs(np.array(shown) - model.csv("test_predictions")["Stacking 1"].iloc[:8].to_numpy()).max() < 0.5
    assert store.current_day() == model.csv("test_predictions").index[8]


def test_store_rules():
    day = store.current_day()
    assert day == model.test_start()
    assert store.check(None, None) and store.check(1.5, None) and store.check(10, -1) and not store.check(10, None)
    with pytest.raises(ValueError):
        store.save(model.next_operating_day(day), 100, None, 100, 100)      # cannot skip a day
    store.save(day, 7000, 7500, 7400.0, 7900, -10, "Event")
    assert store.current_day() == model.next_operating_day(day)
    store.set_note(day, "Campus event")
    assert store.load()["Note"].iloc[0] == "Campus event"
    assert store.undo() == day and store.load().empty
    store.save(day, 7000, None, 7400.0, 7900)
    store.reset()
    assert store.load().empty


def test_sundays_skipped_and_calendar_end():
    assert model.next_operating_day(pd.Timestamp("2026-08-08")) == pd.Timestamp("2026-08-10")
    assert model.next_operating_day(model.last_data_day()) == pd.Timestamp("2027-01-01")
    assert model.period_of(pd.Timestamp("2027-01-02")) is None
    cond = model.conditions(pd.Timestamp("2027-01-02"), model.history()[model.meta()["target"]].astype(float), period="Semester Break")
    assert cond["Academic_Period"] == "Semester Break" and cond["Weekend_Status"] == "Weekend"


def test_week_ahead():
    sales = store.sales()
    wa = model.week_ahead(model.test_start(), sales, 6)
    assert len(wa) == 6 and (wa["Date"].dt.dayofweek < 6).all()
    assert abs(wa["Forecast"].iloc[0] - model.csv("test_predictions")["Stacking 1"].iloc[0]) < 0.5


def test_explanation_adds_up():
    cond = model.conditions(model.test_start(), store.sales())
    base, contrib = model.explain(cond)
    assert abs(base + contrib.sum() - model.forecast(cond)) < 1


def test_plan_and_outcomes():
    e = insights.errors()
    assert insights.plan(5000, 0.8) == int(np.ceil((5000 + np.quantile(e, 0.8)) / 10) * 10)
    assert insights.plan(5000, 0.8, -30) < insights.plan(5000, 0.8)
    log = pd.DataFrame({"Date": pd.to_datetime(["2026-08-07", "2026-08-08"]), "Period": ["Teaching"] * 2,
                        "Forecast": [7000.0, 4000.0], "Plan": [7500, 4400], "Adjustment (%)": [0, 0], "Note": ["", ""],
                        "Prepared": [7600, None], "Sold": [7200, 4600]})
    out = insights.outcomes(log)
    assert list(out["Leftover"]) == [400, 0] and list(out["Short"]) == [0, 200]
    assert "400 meals were left over" in insights.compare(out.iloc[0])
    assert "200 meals short" in insights.compare(out.iloc[1])


def test_reason_wording():
    cond = {"Day_of_Week": "Saturday", "Weekend_Status": "Weekend", "Academic_Period": "Semester Break", "Month": "September",
            "Price_per_meal": 10.0}
    contrib = pd.Series({"Weekend_Status": -2100.0, "Day_of_Week": -340.0, "Academic_Period": -1890.0, "Month": 30.0})
    s = insights.reason(2030, 6420, contrib, cond)
    assert "it is a Saturday (−2,450)" in s and "it is the semester break (−1,900)" in s and "below a typical day" in s


def fake_log(diffs, adjustment=0, note="", prepared_gap=None, start="2026-08-10"):
    # A log of ordinary weekdays where sales differ from a 7,000-meal forecast by the given amounts
    days = pd.bdate_range(start, periods=len(diffs))
    fc = np.full(len(diffs), 7000.0)
    sold = (fc * (1 + adjustment / 100) + np.array(diffs)).astype(int)
    prepared = None if prepared_gap is None else (sold - np.array(prepared_gap)).astype(float)
    return pd.DataFrame({"Date": days, "Period": "Teaching", "Forecast": fc, "Plan": 7500, "Adjustment (%)": adjustment,
                         "Note": note, "Prepared": prepared if prepared is not None else np.nan, "Sold": sold})


def titles(log, level=0.8):
    return [a[1] for a in insights.alerts(insights.outcomes(log), level)]


def test_limits_from_past_errors():
    limits = insights.unusual_limits()
    assert limits[True][0] < -1000 and limits[True][1] > 1100 and limits[False][1] < limits[True][1]
    size, lean = insights.window_limits()
    assert 700 < size < 850 and 300 < lean < 420
    assert [insights.short_limit(x) for x in (0.5, 0.8, 0.95)] == [None, 4, 2]


def test_unusual_and_explained_days():
    log = fake_log([0, 50, -1500])
    assert titles(log)[0].startswith("Unusual day")
    log.loc[2, "Note"] = "Fire drill"
    assert titles(log)[0].startswith("Explained")
    assert not [t for t in titles(log) if t.startswith("Unusual")]


def test_adjusted_days_compared_with_adjusted_forecast():
    out = insights.outcomes(fake_log([0, 30, -40], adjustment=-20, note="Exam week"))
    assert not out["Unusual"].any() and not out["Counted"].any()
    assert "than the adjusted forecast" in insights.compare(out.iloc[0])


def test_accuracy_drift_and_on_track():
    rng = np.random.default_rng(0)
    assert "Forecasts are on track" in titles(fake_log(rng.normal(0, 300, 14)))
    assert "Forecasts have been too high lately" in titles(fake_log(rng.normal(-450, 150, 14)))
    assert "Forecasts have been too low lately" in titles(fake_log(rng.normal(450, 150, 14)))
    big = fake_log(np.tile([1000, -1000], 7))
    assert "Forecasts need updating" in titles(big)


def test_running_short_warning():
    short4 = fake_log([0] * 7, prepared_gap=[100, 100, 100, 100, -100, -100, -100])
    short3 = fake_log([0] * 7, prepared_gap=[100, 100, 100, -100, -100, -100, -100])
    assert "Food keeps running short" in titles(short4, 0.8)
    assert "Food keeps running short" not in titles(short3, 0.8)
    assert "Food keeps running short" in titles(short3, 0.95)
    assert "Food keeps running short" not in titles(short4, 0.5)


def test_closed_days():
    first = store.current_day()
    store.close(first, "Public holiday")
    assert store.current_day() == model.next_operating_day(first)
    with pytest.raises(ValueError):
        store.close(pd.Timestamp("2026-08-09"))                      # a Sunday
    wa = model.week_ahead(store.current_day(), store.sales(), 6, closed=[pd.Timestamp("2026-08-12")])
    assert pd.Timestamp("2026-08-12") not in set(wa["Date"])
    store.reopen(first)
    assert store.current_day() == first
    day = store.current_day()
    store.save(day, 7000, None, 7000.0, 7500)
    with pytest.raises(ValueError):
        store.close(day)                                             # already recorded


def test_corrections():
    day = store.current_day()
    store.save(day, 750, 8000, 7400.0, 7900)
    store.correct(day, 7500, 8000)
    row = store.load().iloc[0]
    assert row["Sold"] == 7500 and "sold corrected from 750" in row["Note"]
    with pytest.raises(ValueError):
        store.correct(day, -5, None)


def test_special_day_suggestion():
    log = fake_log([0, 0, 0], adjustment=-20, note="Exam week")
    log["Sold"] = [5700, 5600, 5650]
    count, effect = insights.past_effect(log, "Exam week")
    assert count == 3 and -21 < effect < -18
    assert insights.nearest_level(effect) == "Fewer than usual"
    assert insights.past_effect(log, "Campus event") == (0, None)


def test_prep_sheet():
    wa = model.week_ahead(model.test_start(), store.sales(), 6)
    html = insights.prep_sheet(model.test_start(), "Teaching", 7930, 7459, 6690, 8150, "Because.", 0.8, -10, "Event", wa.iloc[1:])
    assert "7,930 meals" in html and "Friday 07 August 2026" in html and "-10 %" in html


# ---------- Pages ----------

STAFF_PAGES = ["pages/today.py", "pages/records.py", "pages/what_if.py"]
JARGON = r"\b(SHAP|LIME|MAE|RMSE|SMAPE|R²|R2|Ridge|regression|algorithm|lag|lags|percentile|out-of-fold|stacking|model|XGBoost|SVR|dataset)\b"


def app_test():
    from streamlit.testing.v1 import AppTest
    return AppTest.from_file(str(HERE / "app.py"), default_timeout=180)


def page_text(at):
    # All text the viewer sees on the page, sidebar included
    parts = []
    for kind in ["markdown", "caption", "title", "header", "subheader", "info", "warning", "success", "error", "button",
                 "number_input", "selectbox", "text_input", "slider", "expander", "radio", "date_input"]:
        for el in getattr(at, kind, []):
            for attr in ["value", "label", "body", "help", "placeholder"]:
                v = getattr(el, attr, None)
                if isinstance(v, str):
                    parts.append(v)
    return " ".join(parts)


def test_staff_pages_load_without_jargon():
    record_recorded_days(8)
    for page in STAFF_PAGES:
        at = app_test()
        at.run()
        if page != STAFF_PAGES[0]:
            at.switch_page(page).run()
        assert not at.exception, (page, at.exception)
        found = re.findall(JARGON, page_text(at), flags=re.IGNORECASE)
        assert not found, (page, sorted(set(found)))


def test_performance_page_loads():
    at = app_test()
    at.run()
    at.switch_page("pages/performance.py").run()
    assert not at.exception
    at.button(key="lime").click().run()
    assert not at.exception


def test_record_through_the_page():
    at = app_test()
    at.run()
    assert not [b for b in at.button if b.key == "demo_fill"]          # hidden unless demo mode is on
    at.toggle(key="demo").set_value(True).run()
    at.button(key="demo_fill").click().run()
    at.get("form_submit_button")[0].click().run()
    assert not at.exception
    assert store.current_day() == pd.Timestamp("2026-08-08")
    at.button(key="undo").click().run()
    assert store.load().empty
