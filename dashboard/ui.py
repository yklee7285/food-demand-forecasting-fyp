"""Shared look: styles, cards, notices and charts used by the pages."""
from html import escape

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

BLUE, ORANGE, GREY, MUTED = "#1d4ed8", "#c2410c", "#9ca3af", "#4b5563"

CSS = """<style>
.block-container{max-width:1080px;padding-top:2rem}
.eyebrow{font-size:.8rem;font-weight:600;color:#1d4ed8;letter-spacing:.05em;text-transform:uppercase;margin-bottom:-.6rem}
.hero{background:#1d4ed8;color:#fff;border-radius:12px;padding:20px 22px;min-height:128px;box-sizing:border-box;margin-bottom:16px}
.hero .label{font-size:.95rem}.hero .value{font-size:2.9rem;font-weight:700;line-height:1.05}.hero .sub{font-size:.85rem}
.stat{background:#fff;border:1px solid #e5e7eb;border-radius:12px;padding:16px 18px;min-height:128px;box-sizing:border-box;margin-bottom:16px}
.stat .label{font-size:.85rem;color:#4b5563}.stat .value{font-size:1.9rem;font-weight:700;line-height:1.15;color:#1f2937}
.stat .sub{font-size:.8rem;color:#4b5563}
.notice{display:flex;gap:12px;align-items:flex-start;border-radius:12px;padding:12px 16px;margin:0 0 12px;line-height:1.5;font-size:.95rem}
.notice .icon{flex:none;width:26px;height:26px;border-radius:999px;display:flex;align-items:center;justify-content:center;font-weight:700}
.notice.ok{background:#fff;border:1px solid #e5e7eb}.notice.ok .icon{background:#e8efff;color:#1d4ed8}
.notice.warning{background:#fff7ed;border:1px solid #fdba74}.notice.warning .icon{background:#c2410c;color:#fff}
.notice.explained{background:#f1f5f9;border:1px solid #e2e8f0}.notice.explained .icon{background:#64748b;color:#fff}
.day{border:1px solid #e5e7eb;border-radius:10px;padding:10px 12px;background:#fff;margin-bottom:8px}
.day.today{border:2px solid #1d4ed8}.day .d{font-size:.8rem;font-weight:600}.day .v{font-size:1.3rem;font-weight:700}
.day .p{font-size:.75rem;color:#4b5563}
.stButton button,.stDownloadButton button,.stFormSubmitButton button{min-height:44px}
</style>"""


def style():
    st.markdown(CSS, unsafe_allow_html=True)


def n(v):
    return f"{v:,.0f}"


def header(eyebrow, title, sub=None):
    st.markdown(f'<div class="eyebrow">{escape(eyebrow)}</div>', unsafe_allow_html=True)
    st.title(title, anchor=False)
    if sub:
        st.caption(sub)


def hero(label, value, sub):
    st.markdown(f'<div class="hero"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div>'
                f'<div class="sub">{escape(sub)}</div></div>', unsafe_allow_html=True)


def stat(label, value, sub=""):
    st.markdown(f'<div class="stat"><div class="label">{escape(label)}</div><div class="value">{escape(value)}</div>'
                f'<div class="sub">{escape(sub)}</div></div>', unsafe_allow_html=True)


def notice(kind, title, text):
    icon = {"warning": "!", "explained": "i"}.get(kind, "✓")
    st.markdown(f'<div class="notice {kind}"><span class="icon" aria-hidden="true">{icon}</span>'
                f'<div><b>{escape(title)}</b> {escape(text)}</div></div>', unsafe_allow_html=True)


def outlook(frame, today):
    # Row of small day cards for the week ahead
    cols = st.columns(len(frame))
    for col, r in zip(cols, frame.itertuples()):
        cls = "day today" if r.Date == today else "day"
        tag = "Today" if r.Date == today else r.Period
        col.markdown(f'<div class="{cls}"><div class="d">{r.Date:%a %d}</div><div class="v">{n(round(r.Forecast, -1))}</div>'
                     f'<div class="p">{escape(tag)}</div></div>', unsafe_allow_html=True)


def bar_height(bars):
    # Plot height that gives each bar and its label enough room
    return 36 * bars


def chart(spec, height):
    # Draw a chart at a fixed height in pixels (plot plus title, axis and legend), so it never gets squeezed
    st.altair_chart(spec, width="stretch", height=height)


def contribution_chart(contrib, cond, names, title):
    # Horizontal bars: how much each factor pushed the number up or down, in meals
    def label(var):
        v = cond.get(var)
        if isinstance(v, str):
            return f"{names[var]}: {v}"
        if var == "Price_per_meal":
            return f"{names[var]}: RM {v:.2f}"
        return f"{names[var]}: {n(v)} meals"
    data = pd.DataFrame({"Meals": contrib.values, "Factor": [f"{label(k)}  ({v:+,.0f})" for k, v in contrib.items()]})
    data = data.reindex(data.Meals.abs().sort_values(ascending=False).index)
    data["Effect"] = np.where(data.Meals >= 0, "Raises the number", "Lowers the number")
    return alt.Chart(data).mark_bar().encode(
        x=alt.X("Meals:Q", title="Meals, compared with a typical day"),
        y=alt.Y("Factor:N", sort=list(data.Factor), title=None, axis=alt.Axis(labelLimit=380, labelOverlap=False)),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises the number", "Lowers the number"], range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["Factor", alt.Tooltip("Meals:Q", format="+,.0f")]).properties(title=title, height=bar_height(len(data)))


def sold_vs_forecast(frame, extra=None):
    # Line chart of meals sold and forecast per day, with the likely range as a shaded band when given
    frame = frame.copy()
    if len(frame) <= 31:
        # A few days: one label per day
        frame["Day"] = frame["Date"].dt.strftime("%a %d %b")
        x = alt.X("Day:O", sort=list(frame["Day"]), title=None, axis=alt.Axis(labelAngle=0 if len(frame) <= 10 else -45))
    else:
        x = alt.X("Date:T", title=None, axis=alt.Axis(format="%d %b", tickMinStep=86_400_000, labelOverlap=True))
    when = alt.Tooltip("Date:T", title="Day", format="%a %d %b %Y")
    layers = []
    if "Low" in frame:
        layers.append(alt.Chart(frame.reset_index(drop=True)).mark_area(opacity=0.15, color=BLUE).encode(
            x=x, y=alt.Y("Low:Q", title="Meals"), y2="High:Q",
            tooltip=[when, alt.Tooltip("Low:Q", title="Expected range from", format=",.0f"),
                     alt.Tooltip("High:Q", title="Expected range to", format=",.0f")]))
    series = ["Meals sold", "Forecast"] + ([extra] if extra else [])
    long = frame.melt([c for c in ("Date", "Day") if c in frame], value_vars=series, var_name="Series", value_name="Meals")
    layers.append(alt.Chart(long).mark_line(point=len(frame) < 40).encode(
        x=x, y=alt.Y("Meals:Q", title="Meals"),
        color=alt.Color("Series:N", scale=alt.Scale(domain=series, range=[BLUE, ORANGE, GREY][:len(series)]),
                        legend=alt.Legend(orient="bottom", title=None, labelLimit=300)),
        strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=series, range=[[1, 0], [6, 3], [2, 2]][:len(series)]),
                                  legend=None),
        tooltip=[when, alt.Tooltip("Series:N", title="Line"), alt.Tooltip("Meals:Q", format=",.0f")]))
    return alt.layer(*layers).properties(height=300)
