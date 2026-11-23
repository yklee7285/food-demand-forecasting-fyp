"""Turns forecasts and recorded days into plain-language advice: reason, plan, comparison, alerts and summaries."""
import math
from html import escape

import numpy as np
import pandas as pd
from scipy.stats import binom

import model

DEFAULT_LEVEL = 0.80        # default chance of having enough food
RANGE = (0.10, 0.90)        # the likely range covers 8 in 10 past days
UNUSUAL = (0.025, 0.975)    # a day outside this wider band (about 1 in 20 days) is flagged as unusual
THRESHOLD = 100             # reasons smaller than this many meals are not mentioned
ALERT_DAYS, ALERT_MIN = 14, 7   # the accuracy checks use the last 14 ordinary days, from 7 onwards
SHORT_DAYS = 7                  # the running-short check uses the last 7 ordinary days

NAMES = {"Academic_Period": "Academic period", "Weekend_Status": "Weekend or weekday", "Day_of_Week": "Day of the week",
         "Month": "Month", "Price_per_meal": "Meal price", "Quantity_last_day": "Sales yesterday",
         "Quantity_last_two_days": "Sales over the last two days", "Quantity_last_week": "Sales over the past week"}
PERIOD_TEXT = {"Teaching": "it is a teaching week", "Mid-Semester Break": "it is the mid-semester break",
               "Semester Break": "it is the semester break"}


def rnd(v, step=50):
    return int(round(v / step) * step)


def n(v):
    # Whole number with thousands separators
    return f"{v:,.0f}"


def signed(v):
    r = rnd(v)
    return f"{'+' if r > 0 else '−' if r < 0 else ''}{abs(r):,}"


# ---------- Plan for a day ----------

def errors():
    # Past forecast errors (meals sold minus forecast) on the training days, each made without seeing that day
    return model.csv("oof_residuals")["Error"].to_numpy()


def likely_range(fc):
    lo, hi = np.quantile(errors(), RANGE)
    return fc + lo, fc + hi


def plan(fc, level=DEFAULT_LEVEL, adjustment=0):
    # Meals to prepare: forecast plus a safety margin for the chosen level, then any special-day adjustment, rounded up to 10
    base = fc + np.quantile(errors(), level)
    return int(math.ceil(base * (1 + adjustment / 100) / 10) * 10)


def reason(fc, base, contrib, cond):
    # One-sentence reason: how far from a typical day and the main causes
    c = contrib.drop(["Weekend_Status", "Day_of_Week"], errors="ignore")
    c["Day"] = contrib.get("Weekend_Status", 0.0) + contrib.get("Day_of_Week", 0.0)
    c = c.reindex(c.abs().sort_values(ascending=False).index)

    def phrase(var, v):
        if var == "Day":
            return f"it is a {cond['Day_of_Week']}"
        if var == "Academic_Period":
            return PERIOD_TEXT[cond["Academic_Period"]]
        if var == "Month":
            return f"it is {cond['Month']}"
        if var == "Price_per_meal":
            return f"the meal price (RM {cond['Price_per_meal']:.2f})"
        return f"{NAMES[var].lower()} {'were' if 'Sales' in NAMES[var] else 'was'} {'higher' if v > 0 else 'lower'} than usual"

    found = [f"{phrase(k, v)} ({signed(v)})" for k, v in c.items() if abs(v) >= THRESHOLD][:3]
    diff = fc - base
    if not found or abs(diff) < THRESHOLD:
        return f"Close to a typical day (about {n(rnd(base))} meals)."
    listed = found[0] if len(found) == 1 else ", ".join(found[:-1]) + " and " + found[-1]
    return (f"About {n(abs(rnd(diff)))} {'above' if diff > 0 else 'below'} a typical day "
            f"(about {n(rnd(base))} meals), mainly because {listed}.")


# ---------- Special days ----------

LEVELS = {"Much fewer than usual": -40, "Fewer than usual": -20, "A little fewer than usual": -10,
          "A little more than usual": 10, "More than usual": 20, "Much more than usual": 40}
REASONS = ["Exam week", "Campus event", "Heavy rain", "Event near campus", "Other"]


def past_effect(log, reason):
    # How sales compared with the forecast on earlier days with the same reason: (days, average % difference)
    same = log[(log["Note"].str.split(";").str[0].str.strip().str.lower() == reason.strip().lower()) & (log["Adjustment (%)"] != 0)]
    if len(same) < 2:
        return len(same), None
    return len(same), float(((same["Sold"] / same["Forecast"]) - 1).mean() * 100)


def nearest_level(pct):
    # The preset closest to a percentage change
    return min(LEVELS, key=lambda k: abs(LEVELS[k] - pct))


# ---------- Recorded days ----------

def _busy(dates):
    # True for busy days (teaching week, Monday to Friday), False for quiet days (Saturdays and breaks)
    hist = model.history()
    period = pd.Series([hist["Academic_Period"].get(d, None) for d in dates], index=dates)
    return (pd.DatetimeIndex(dates).dayofweek < 5) & (period.values == "Teaching")


def unusual_limits():
    # Lower and upper limits of a normal miss for busy and quiet days (2.5th and 97.5th percentiles of past errors)
    oof = model.csv("oof_residuals")
    busy = _busy(oof.index)
    return {True: tuple(np.quantile(oof["Error"][busy], UNUSUAL)), False: tuple(np.quantile(oof["Error"][~busy], UNUSUAL))}


def window_limits():
    # Limits for the 14-day checks: average miss and average lean, each worse than 99 % of past 14-day stretches
    e = model.csv("oof_residuals")["Error"]
    return (float(e.abs().rolling(ALERT_DAYS).mean().quantile(0.99)),
            float(e.rolling(ALERT_DAYS).mean().abs().quantile(0.99)))


def outcomes(log):
    # Recorded days with the expected sales (after any adjustment), likely range, leftovers, shortfall and checks
    cols = ["Expected", "Low", "High", "Used", "Leftover", "Short", "Value", "Difference", "Unusual", "Explained", "Counted"]
    if log.empty:
        return log.assign(**{c: [] for c in cols})
    out = log.copy()
    out["Expected"] = out["Forecast"] * (1 + out["Adjustment (%)"] / 100)
    lo, hi = np.quantile(errors(), RANGE)
    out["Low"], out["High"] = out["Expected"] + lo, out["Expected"] + hi
    out["Used"] = out["Prepared"].fillna(out["Plan"])
    out["Leftover"] = (out["Used"] - out["Sold"]).clip(lower=0)
    out["Short"] = (out["Sold"] - out["Used"]).clip(lower=0)
    out["Value"] = out["Leftover"] * out["Date"].map(model.price_of)
    out["Difference"] = out["Sold"] - out["Expected"]
    limits, busy = unusual_limits(), _busy(out["Date"])
    lower = np.where(busy, limits[True][0], limits[False][0])
    upper = np.where(busy, limits[True][1], limits[False][1])
    out["Unusual"] = (out["Difference"] < lower) | (out["Difference"] > upper)
    out["Explained"] = out["Note"].str.strip() != ""
    out["Counted"] = (out["Adjustment (%)"] == 0) & ~out["Explained"]      # ordinary days, used in the 14-day checks
    return out


def compare(row):
    # How a recorded day went, in plain words
    diff = row["Difference"]
    against = "the adjusted forecast" if row["Adjustment (%)"] else "forecast"
    where = ("inside the expected range" if row["Low"] <= row["Sold"] <= row["High"] else
             "above the expected range" if row["Sold"] > row["High"] else "below the expected range")
    if row["Unusual"]:
        where = "far " + where
    text = f"Sold {n(row['Sold'])} meals, {n(abs(rnd(diff, 10)))} {'more' if diff > 0 else 'fewer'} than {against}, {where}."
    made = "prepared" if pd.notna(row["Prepared"]) else "planned"
    if row["Leftover"] > 0:
        text += f" {n(row['Used'])} {made}, so {n(row['Leftover'])} meals were left over."
    elif row["Short"] > 0:
        text += f" {n(row['Used'])} {made}, so {n(row['Short'])} meals short."
    return text


def training_difference():
    # Average size of the past forecast errors on the training days
    return float(np.abs(errors()).mean())


def short_limit(level):
    # Fewest short days out of the last 7 that would happen by chance less than 1 time in 20 at this level
    if level < 0.7:
        return None
    return next((k for k in range(1, SHORT_DAYS + 1) if binom.sf(k - 1, SHORT_DAYS, 1 - level) < 0.05), None)


def alerts(out, level=DEFAULT_LEVEL):
    # Warnings as (kind, title, text): kind "warning" is counted, "explained" and "ok" are not
    found = []
    recent = out.tail(ALERT_DAYS)
    for _, r in recent[recent["Unusual"]].iloc[::-1].iterrows():
        side = "more" if r["Difference"] > 0 else "fewer"
        what = f"Sold {n(r['Sold'])}, about {n(abs(rnd(r['Difference'])))} {side} than expected."
        if r["Explained"]:
            found.append(("explained", f"Explained: {r['Date']:%a %d %b}", f"{what} Note: {r['Note']}"))
        else:
            found.append(("warning", f"Unusual day: {r['Date']:%a %d %b}",
                          f"{what} That is far outside the expected range. Add a note in the table on the Records page "
                          f"if you know why (for example an event), or correct the number if it was typed wrongly."))
    ordinary = out[out["Counted"]]
    last7 = ordinary.tail(SHORT_DAYS)
    k = short_limit(level)
    if k and len(last7) == SHORT_DAYS and (last7["Short"] > 0).sum() >= k:
        found.append(("warning", "Food keeps running short",
                      f"Food ran short on {(last7['Short'] > 0).sum()} of the last {SHORT_DAYS} ordinary days, more than "
                      f"expected. Consider raising 'Chance of having enough food' in the sidebar."))
    window = ordinary.tail(ALERT_DAYS)
    if len(window) >= ALERT_MIN:
        size_limit, lean_limit = window_limits()
        avg, lean, usual = window["Difference"].abs().mean(), window["Difference"].mean(), training_difference()
        days = f"Over the last {len(window)} ordinary days"
        if avg > size_limit:
            found.append(("warning", "Forecasts need updating",
                          f"{days} the forecasts were off by {n(avg)} meals on average, much more than usual "
                          f"(about {n(usual)}). Ask for the forecasts to be updated with the new sales."))
        if abs(lean) > lean_limit:
            direction = "too high" if lean < 0 else "too low"
            more = "fewer" if lean < 0 else "more"
            found.append(("warning", f"Forecasts have been {direction} lately",
                          f"{days} the canteen sold about {n(abs(rnd(lean, 10)))} meals a day {more} than forecast. "
                          f"Keep this in mind when preparing, and ask for the forecasts to be updated with the new sales."))
        if avg <= size_limit and abs(lean) <= lean_limit:
            found.append(("ok", "Forecasts are on track",
                          f"{days} the forecasts were off by {n(avg)} meals on average, in line with the usual {n(usual)}."))
    return found


def period_rows(out, period):
    # Recorded days in "This week", "This month" or "All recorded" (relative to the last recorded day)
    if out.empty or period == "All recorded":
        return out
    last = out["Date"].iloc[-1]
    if period == "This week":
        return out[out["Date"] > last - pd.Timedelta(days=last.dayofweek + 1)]
    return out[(out["Date"].dt.year == last.year) & (out["Date"].dt.month == last.month)]


def summary(rows):
    # Totals for a set of recorded days, with what preparing the same amount every day would have given
    fixed = round(model.csv("oof_residuals")["Actual"].mean())
    return {"days": len(rows), "enough": int((rows["Short"] == 0).sum()), "leftover": rows["Leftover"].sum(),
            "value": rows["Value"].sum(), "short": rows["Short"].sum(), "short_days": int((rows["Short"] > 0).sum()),
            "difference": rows["Difference"].abs().mean() if len(rows) else 0.0, "fixed": fixed,
            "fixed_leftover": (fixed - rows["Sold"]).clip(lower=0).sum(), "fixed_short": (rows["Sold"] - fixed).clip(lower=0).sum()}


# ---------- Test period (for the Performance page) ----------

def backtest(level=DEFAULT_LEVEL):
    # Leftovers and shortfall on the test days for the plan, the plain forecast and a fixed daily amount
    tp = model.csv("test_predictions")
    sold, fc = tp["Actual"].to_numpy(), tp["Stacking 1"].to_numpy()
    fixed = round(model.csv("oof_residuals")["Actual"].mean())
    rules = {f"Plan ({level:.0%} chance of enough)": np.array([plan(f, level) for f in fc]),
             "Plain forecast": np.round(fc), "Same amount every day": np.full(len(sold), fixed)}
    return pd.DataFrame({k: {"Leftover meals": np.clip(p - sold, 0, None).sum(), "Meals short": np.clip(sold - p, 0, None).sum(),
                             "Days with enough food": int((p >= sold).sum())} for k, p in rules.items()}).T


# ---------- Prep sheet ----------

def prep_sheet(day, period, prepare, fc, lo, hi, why, level, adjustment, note, outlook):
    # A print-ready A4 page for the kitchen, as HTML
    adj = f"{adjustment:+d} % ({escape(note) or 'no reason given'})" if adjustment else "None"
    cells = "".join(f"<td><b>{r.Date:%a %d}</b><br>{n(r.Forecast)}</td>" for r in outlook.itertuples())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Prep sheet {day:%Y-%m-%d}</title>
<style>@page{{size:A4;margin:18mm}}body{{font-family:Arial,Helvetica,sans-serif;color:#111;max-width:760px;margin:24px auto}}
h1{{margin:0;font-size:26px}}.box{{border:2px solid #111;border-radius:10px;padding:20px;margin:22px 0}}
.big{{font-size:60px;font-weight:700;line-height:1}}table{{border-collapse:collapse;width:100%}}
td{{border:1px solid #888;padding:8px;font-size:14px}}.lines div{{margin:22px 0}}small{{color:#333}}</style></head><body>
<div style="border-bottom:2px solid #111;padding-bottom:12px"><div style="font-size:13px;font-weight:700;letter-spacing:.05em">KITCHEN PREP SHEET</div>
<h1>{day:%A %d %B %Y}</h1><div>{escape(period)}</div></div>
<div class="box"><div style="font-weight:700">Prepare</div><div class="big">{n(prepare)} meals</div>
<div style="margin-top:8px">Expected to sell {n(fc)} · likely {n(lo)} – {n(hi)} · enough food on about {level:.0%} of days</div></div>
<p><b>Why:</b> {escape(why)}</p><p><b>Special-day adjustment:</b> {adj}</p>
<p><b>Rest of the week (rough guide)</b></p><table><tr>{cells}</tr></table>
<div class="lines"><div>Meals prepared: ____________ &nbsp;&nbsp; Meals sold: ____________</div><div>Checked by: ____________ &nbsp;&nbsp; Notes: ______________________</div></div>
<small>The plan is a suggestion based on past sales. Demo data: the sales are synthetic.</small></body></html>"""
