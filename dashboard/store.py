"""Daily log of meals sold and prepared (data/daily_log.csv) and the days the canteen is closed (data/closed_days.csv)."""
import os
from pathlib import Path

import pandas as pd

import model

DATA = Path(__file__).resolve().parent / "data"
LOG = Path(os.environ.get("DASHBOARD_LOG", DATA / "daily_log.csv"))
CLOSED = Path(os.environ.get("DASHBOARD_CLOSED", DATA / "closed_days.csv"))
COLUMNS = ["Date", "Period", "Forecast", "Plan", "Adjustment (%)", "Note", "Prepared", "Sold"]
MAX_MEALS = 50_000      # entries above this are refused as typing mistakes


# ---------- Recorded days ----------

def load():
    # All recorded days, oldest first
    if not LOG.exists():
        return pd.DataFrame(columns=COLUMNS)
    log = pd.read_csv(LOG, parse_dates=["Date"])
    log["Note"] = log["Note"].fillna("").astype(str)
    log["Prepared"] = pd.to_numeric(log["Prepared"], errors="coerce")
    log["Adjustment (%)"] = log["Adjustment (%)"].fillna(0).astype(int)
    return log.sort_values("Date").reset_index(drop=True)


def _write(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, date_format="%Y-%m-%d")


def sales():
    # Meals sold per open day: the dataset before the log starts, then the recorded days
    hist = model.history()
    before = hist.loc[hist.index < model.test_start(), model.meta()["target"]].astype(float)
    log = load()
    return pd.concat([before, pd.Series(log["Sold"].astype(float).values, index=pd.DatetimeIndex(log["Date"]))])


def check(sold, prepared):
    # Problems with an entry, in plain words (empty if it can be saved)
    problems = []
    if sold is None or pd.isna(sold):
        problems.append("Enter the number of meals sold.")
    elif not float(sold).is_integer() or not 0 <= sold <= MAX_MEALS:
        problems.append(f"Meals sold must be a whole number from 0 to {MAX_MEALS:,}.")
    if prepared is not None and not pd.isna(prepared) and (not float(prepared).is_integer() or not 0 <= prepared <= MAX_MEALS):
        problems.append(f"Meals prepared must be a whole number from 0 to {MAX_MEALS:,}.")
    return problems


def save(day, sold, prepared, forecast, plan, adjustment=0, note="", period=None):
    # Record the current day; other days cannot be recorded, so none are skipped or entered twice
    day = pd.Timestamp(day)
    if day != current_day():
        raise ValueError(f"Only {current_day():%A %d %B %Y} can be recorded now.")
    problems = check(sold, prepared)
    if problems:
        raise ValueError(" ".join(problems))
    row = {"Date": day, "Period": period or model.period_of(day), "Forecast": round(forecast, 1), "Plan": int(plan),
           "Adjustment (%)": int(adjustment), "Note": note, "Prepared": None if prepared is None else int(prepared),
           "Sold": int(sold)}
    log = load()
    _write(pd.DataFrame([row], columns=COLUMNS) if log.empty else pd.concat([log, pd.DataFrame([row])], ignore_index=True), LOG)


def set_note(day, note):
    # Add or change the note of a recorded day
    log = load()
    log.loc[log["Date"] == pd.Timestamp(day), "Note"] = note.strip()
    _write(log, LOG)


def correct(day, sold, prepared):
    # Correct the meals sold or prepared of a recorded day; the note keeps the old values
    problems = check(sold, prepared)
    if problems:
        raise ValueError(" ".join(problems))
    log = load()
    i = log.index[log["Date"] == pd.Timestamp(day)][0]
    old_sold, old_prep = log.at[i, "Sold"], log.at[i, "Prepared"]
    changes = []
    if int(sold) != old_sold:
        changes.append(f"sold corrected from {old_sold:,}")
    new_prep = None if prepared is None or pd.isna(prepared) else int(prepared)
    if (new_prep is None) != pd.isna(old_prep) or (new_prep is not None and new_prep != old_prep):
        changes.append(f"prepared corrected from {'blank' if pd.isna(old_prep) else f'{int(old_prep):,}'}")
    if not changes:
        return
    log.at[i, "Sold"], log.at[i, "Prepared"] = int(sold), new_prep
    note = log.at[i, "Note"]
    log.at[i, "Note"] = "; ".join([n for n in [note, ", ".join(changes)] if n])
    _write(log, LOG)


def undo():
    # Remove the last recorded day; returns its date, or None if nothing was recorded
    log = load()
    if log.empty:
        return None
    _write(log.iloc[:-1], LOG)
    return log["Date"].iloc[-1]


# ---------- Closed days ----------

def closed_days():
    # Days the canteen is closed (for example public holidays), with the reason
    if not CLOSED.exists():
        return pd.DataFrame(columns=["Date", "Reason"])
    frame = pd.read_csv(CLOSED, parse_dates=["Date"])
    frame["Reason"] = frame["Reason"].fillna("").astype(str)
    return frame.sort_values("Date").reset_index(drop=True)


def close(day, reason=""):
    # Mark a day as closed; recorded days and Sundays cannot be closed
    day = pd.Timestamp(day)
    log = load()
    if day.dayofweek == 6:
        raise ValueError("The canteen is already closed on Sundays.")
    if not log.empty and day <= log["Date"].iloc[-1]:
        raise ValueError("Days that are already recorded cannot be marked as closed.")
    frame = closed_days()
    frame = frame[frame["Date"] != day]
    _write(pd.concat([frame, pd.DataFrame([{"Date": day, "Reason": reason.strip()}])], ignore_index=True), CLOSED)


def reopen(day):
    # Remove a day from the closed days
    frame = closed_days()
    _write(frame[frame["Date"] != pd.Timestamp(day)], CLOSED)


# ---------- Today ----------

def current_day():
    # The next open day to plan and record: the first test day, or the open day after the last recorded one
    log, closed = load(), closed_days()["Date"]
    if log.empty:
        day = model.test_start()
        return model.next_operating_day(day - pd.Timedelta(days=1), closed) if day in set(closed) else day
    return model.next_operating_day(log["Date"].iloc[-1], closed)


def reset():
    # Delete every recorded day and closed day
    for path in (LOG, CLOSED):
        if path.exists():
            path.unlink()
