"""Today: how many meals to prepare, why, and the end-of-day record."""
import streamlit as st

import insights
import model
import store
import ui


@st.cache_data(show_spinner=False)
def explain(cond_items):
    return model.explain(dict(cond_items))


day = store.current_day()
level = st.session_state.get("level", 80)
raw_log = store.load()
log = insights.outcomes(raw_log)
closed = store.closed_days()

ui.header("Today", f"{day:%A %d %B %Y}")
period = model.period_of(day)
if period is None:
    period = st.selectbox("This day is after the saved calendar. Which academic period is it?", model.PERIODS,
                          key=f"period_{day.date()}")
else:
    st.caption(period)

if not log.empty:
    last = log.iloc[-1]
    kind = "warning" if last["Unusual"] and not last["Explained"] else "ok"
    ui.notice(kind, f"Last recorded, {last['Date']:%a %d %b}:", insights.compare(last))

sales = store.sales()
cond = model.conditions(day, sales, period=period)
fc = model.forecast(cond)
adj_key = f"adj_{day.date()}"
adjustment, note = st.session_state.get(adj_key, (0, ""))
expected = fc * (1 + adjustment / 100)
lo, hi = insights.likely_range(expected)
prepare = insights.plan(fc, level / 100, adjustment)

with st.container(border=True):
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.hero("Prepare", ui.n(prepare), f"meals · enough on about {level} % of days")
    with c2:
        ui.stat("Expected to sell", ui.n(insights.rnd(expected, 10)),
                f"meals · usual day: {ui.n(insights.rnd(fc, 10))}" if adjustment else "meals")
    with c3:
        ui.stat("Likely range", f"{ui.n(insights.rnd(lo, 10))} – {ui.n(insights.rnd(hi, 10))}", "8 in 10 days fall in this range")
    with st.spinner("Working out the reasons..."):
        base, contrib = explain(tuple(cond.items()))
    why = insights.reason(fc, base, contrib, cond)
    st.markdown(f"**Why:** {why}")
    if adjustment:
        st.markdown(f"**Special day:** {note or 'no reason given'}, so the plan expects "
                    f"{insights.nearest_level(adjustment).lower()} ({adjustment:+d} %).")

    with st.expander("Special day? Adjust today's plan"):
        st.caption("Use this when you expect more or fewer customers than a normal day, for example during exam week, "
                   "a campus event or heavy rain.")
        a1, a2 = st.columns(2)
        reason = a1.selectbox("What is happening?", insights.REASONS, key=f"reason_{day.date()}")
        if reason == "Other":
            reason = a1.text_input("Describe it", key=f"other_{day.date()}").strip() or "Other"
        count, effect = insights.past_effect(raw_log, reason)
        options = list(insights.LEVELS)
        default = options.index(insights.nearest_level(effect)) if effect is not None else 1
        choice = a2.radio("How many customers do you expect?", options, index=default, key=f"level_{day.date()}_{reason}",
                          format_func=lambda k: f"{k} ({insights.LEVELS[k]:+d} %)".replace("-", "−"))
        if effect is not None:
            st.info(f"On {count} earlier days marked '{reason}', the canteen sold about {abs(effect):.0f} % "
                    f"{'fewer' if effect < 0 else 'more'} than forecast. The closest choice is selected.")
        else:
            st.caption("The choices are rough guides. After a few days with the same reason, the dashboard will "
                       "suggest a choice based on how those days went.")
        b1, b2, _ = st.columns([1, 1, 2])
        if b1.button("Apply", key="apply_adj"):
            st.session_state[adj_key] = (insights.LEVELS[choice], reason)
            st.rerun()
        if b2.button("Remove adjustment", key="clear_adj", disabled=not adjustment):
            st.session_state.pop(adj_key, None)
            st.rerun()

outlook = model.week_ahead(day, sales, 6, period=period, closed=closed["Date"])
sheet = insights.prep_sheet(day, cond["Academic_Period"], prepare, expected, lo, hi, why, level / 100, adjustment, note,
                            outlook.iloc[1:])
st.download_button("Print prep sheet", sheet, file_name=f"prep_sheet_{day:%Y-%m-%d}.html", mime="text/html",
                   help="Downloads a page for the kitchen. Open it and press Ctrl+P to print.")

st.subheader("Week ahead", anchor=False)
ui.outlook(outlook, day)
st.caption("A rough guide: later days use these forecasts in place of sales not yet recorded. Closed days are skipped.")

st.subheader("End of day: record today's sales", anchor=False)
sold_key = f"sold_{day.date()}"
recorded = model.history()[model.meta()["target"]].get(day)
if st.session_state.get("demo") and recorded is not None:
    st.button("Fill in the recorded sales (demo)", key="demo_fill",
              on_click=lambda: st.session_state.update({sold_key: int(recorded)}))
with st.form("record", clear_on_submit=False, border=True):
    f1, f2 = st.columns(2)
    sold = f1.number_input("Meals sold (required)", min_value=0, max_value=store.MAX_MEALS, value=None, step=1,
                           key=sold_key, placeholder="Type the number")
    prepared = f2.number_input("Meals prepared (optional)", min_value=0, max_value=store.MAX_MEALS, value=None, step=1,
                               key=f"prep_{day.date()}", placeholder=f"For example {ui.n(prepare)}")
    saved = st.form_submit_button(f"Save and go to {model.next_operating_day(day, closed['Date']):%a %d %b}", type="primary")
if saved:
    problems = store.check(sold, prepared)
    if problems:
        for p in problems:
            st.error(p)
    else:
        store.save(day, sold, prepared, fc, prepare, adjustment, note, period=cond["Academic_Period"])
        st.session_state["saved_msg"] = f"Saved {day:%a %d %b}: {ui.n(sold)} meals sold."
        st.rerun()
if msg := st.session_state.pop("saved_msg", None):
    st.success(msg)
u1, u2 = st.columns(2)
if not log.empty and u1.button(f"Undo last entry ({log['Date'].iloc[-1]:%a %d %b})", key="undo"):
    gone = store.undo()
    st.session_state["saved_msg"] = f"Removed the entry for {gone:%a %d %b}."
    st.rerun()
with u2.popover("Canteen closed today?"):
    why_closed = st.text_input("Reason (optional)", placeholder="For example a public holiday", key="closed_reason")
    if st.button(f"Mark {day:%a %d %b} as closed", key="close_today"):
        store.close(day, why_closed)
        st.session_state["saved_msg"] = f"{day:%a %d %b} marked as closed. Nothing needs to be recorded for it."
        st.rerun()

with st.expander("Closed days ahead (for example public holidays)"):
    st.caption("Days marked here are skipped in the week ahead and on the prep sheet.")
    c1, c2, c3 = st.columns([1, 2, 1])
    when = c1.date_input("Day", value=None, min_value=day.date(), format="DD/MM/YYYY", key="closed_day")
    what = c2.text_input("Reason", placeholder="For example a public holiday", key="closed_what")
    c3.write("")
    if c3.button("Add", key="add_closed", disabled=when is None):
        try:
            store.close(when, what)
            st.rerun()
        except ValueError as err:
            st.error(str(err))
    upcoming = closed[closed["Date"] >= day]
    for r in upcoming.itertuples():
        r1, r2 = st.columns([4, 1])
        r1.write(f"{r.Date:%a %d %b %Y}" + (f" · {r.Reason}" if r.Reason else ""))
        if r2.button("Remove", key=f"reopen_{r.Date.date()}"):
            store.reopen(r.Date)
            st.rerun()
    if upcoming.empty:
        st.write("No closed days marked.")

with st.expander("What pushed today's number up or down"):
    ui.chart(ui.contribution_chart(contrib, cond, insights.NAMES,
                                   f"Starting from a typical day of about {ui.n(insights.rnd(base))} meals"),
             ui.bar_height(len(contrib)) + 140)
    st.caption("Each bar shows how many meals a factor added or took away on a normal day; any special-day change is "
               "applied on top. Weekend or weekday and the day of the week describe the same thing, so read them together.")
