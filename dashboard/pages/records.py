"""Records: how the recorded days went, alerts and weekly or monthly totals."""
import pandas as pd
import streamlit as st

import insights
import store
import ui

ui.header("Records", "How the forecasts are doing")
out = insights.outcomes(store.load())
if out.empty:
    st.info("No days recorded yet. Record the meals sold at the end of the day on the Today page.")
    st.stop()

period = st.segmented_control("Show", ["This week", "This month", "All recorded"], default="This month", key="period")
rows = insights.period_rows(out, period or "All recorded")
s = insights.summary(rows)

for kind, title, text in insights.alerts(out, st.session_state.get("level", 80) / 100):
    ui.notice(kind, title, text)

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.stat("Days with enough food", f"{s['enough']} of {s['days']}", f"aim: {st.session_state.get('level', 80)} %")
with c2:
    ui.stat("Leftover meals", ui.n(s["leftover"]), f"worth about RM {ui.n(s['value'])}")
with c3:
    ui.stat("Meals short", ui.n(s["short"]), f"on {s['short_days']} day{'s' if s['short_days'] != 1 else ''}")
with c4:
    ui.stat("Average difference", ui.n(s["difference"]), "meals a day off the forecast")

with st.container(border=True):
    st.subheader("Compared with preparing the same amount every day", anchor=False)
    st.table({"": ["Following the plan", f"Same amount every day ({ui.n(s['fixed'])} meals)"],
              "Leftover meals": [ui.n(s["leftover"]), ui.n(s["fixed_leftover"])],
              "Meals short": [ui.n(s["short"]), ui.n(s["fixed_short"])]})
    st.caption("Uses the meals prepared where they were entered, otherwise the plan shown that day.")

with st.container(border=True):
    st.subheader("Meals sold and forecast", anchor=False)
    chart = rows.drop(columns="Forecast").rename(columns={"Sold": "Meals sold", "Expected": "Forecast"})[
        ["Date", "Meals sold", "Forecast", "Low", "High"]]
    ui.chart(ui.sold_vs_forecast(chart), 400)
    st.caption("The shaded band is the expected range for each day. On special days the forecast includes the adjustment.")

with st.container(border=True):
    st.subheader("Recorded days", anchor=False)
    show = rows.assign(Result=[f"{ui.n(r.Leftover)} left over" if r.Leftover else (f"{ui.n(r.Short)} short" if r.Short else "exact")
                               for r in rows.itertuples()])
    show = show[["Date", "Period", "Forecast", "Plan", "Prepared", "Sold", "Result", "Note"]].iloc[::-1]
    edited = st.data_editor(show, hide_index=True, width="stretch", key="log_table",
                            disabled=[c for c in show.columns if c not in ("Sold", "Prepared", "Note")], column_config={
        "Date": st.column_config.DateColumn(format="ddd D MMM YYYY"),
        "Forecast": st.column_config.NumberColumn(format="%,.0f"), "Plan": st.column_config.NumberColumn(format="%,d"),
        "Prepared": st.column_config.NumberColumn(format="%,d", min_value=0, max_value=store.MAX_MEALS, step=1),
        "Sold": st.column_config.NumberColumn(format="%,d", min_value=0, max_value=store.MAX_MEALS, step=1, required=True),
        "Note": st.column_config.TextColumn(help="Click to add or change a note, for example an event or a closure.")})
    for (_, old), (_, new) in zip(show.iterrows(), edited.iterrows()):
        sold_changed = old["Sold"] != new["Sold"]
        prep_changed = not (pd.isna(old["Prepared"]) and pd.isna(new["Prepared"])) and old["Prepared"] != new["Prepared"]
        if sold_changed or prep_changed:
            try:
                store.correct(old["Date"], new["Sold"], new["Prepared"])
                st.rerun()
            except ValueError as err:
                st.error(f"{old['Date']:%a %d %b}: {err}")
        elif (old["Note"] or "") != (new["Note"] or ""):
            store.set_note(old["Date"], new["Note"] or "")
            st.rerun()
    st.caption("Click a cell to add a note or correct meals sold or prepared. Corrections are written into the note. "
               "A day with a note counts as explained.")
    closed = store.closed_days()
    if not closed.empty:
        st.caption("Closed days: " + ", ".join(f"{r.Date:%a %d %b}" + (f" ({r.Reason})" if r.Reason else "")
                                               for r in closed.itertuples()))
    d1, d2 = st.columns([1, 3])
    d1.download_button("Download as CSV", store.load().to_csv(index=False), "daily_log.csv", "text/csv")
    with d2.popover("Reset (delete all recorded days)"):
        st.write("This deletes every recorded day and closed day and starts again from the first day. It cannot be undone.")
        if st.button("Yes, delete everything", type="primary", key="reset"):
            store.reset()
            st.rerun()
