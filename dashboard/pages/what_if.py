"""What-if: compare the forecast under changed conditions with the forecast under real conditions."""
import altair as alt
import pandas as pd
import streamlit as st

import insights
import model
import store
import ui

LAGS = ["Quantity_last_day", "Quantity_last_two_days", "Quantity_last_week"]


@st.cache_data(show_spinner=False)
def explain(cond_items):
    return model.explain(dict(cond_items))


ui.header("What-if", "What if conditions were different?",
          "Change the academic period, the meal price or recent sales and see how the forecast responds.")

today = store.current_day()
day = pd.Timestamp(st.date_input("Day", value=today.date(), min_value=model.test_start().date(), max_value=today.date(),
                                 format="DD/MM/YYYY", help="Today or any recorded day."))
if day.dayofweek == 6:
    st.warning("The canteen is closed on Sundays. Pick another day.")
    st.stop()
log = store.load()
logged = log.loc[log["Date"] == day, "Period"]
period_real = model.period_of(day) or (logged.iloc[0] if len(logged) else st.session_state.get(f"period_{today.date()}"))
real = model.conditions(day, store.sales(), period=period_real)
hist = model.history()
pmin, pmax = float(hist["Price_per_meal"].min()), float(hist["Price_per_meal"].max())
k = f"wi_{day.date()}"


def reset():
    # Put every control back to the real conditions of the chosen day
    st.session_state[f"{k}_period"] = real["Academic_Period"]
    st.session_state[f"{k}_price"] = min(max(round(real["Price_per_meal"], 2), pmin), pmax)
    for c in LAGS:
        st.session_state[f"{k}_{c}"] = float(min(max(round(real[c]), hist[c].min()), hist[c].max()))


if f"{k}_period" not in st.session_state:
    reset()

left, right = st.columns([1, 2], gap="large")
with left, st.container(border=True):
    st.subheader("Change the conditions", anchor=False)
    period = st.selectbox("Academic period", model.PERIODS, key=f"{k}_period")
    price = st.number_input(f"Average meal price (RM {pmin:.2f} to {pmax:.2f})", pmin, pmax, step=0.5, key=f"{k}_price")
    with st.expander("Recent sales (meals)"):
        lags = {c: st.number_input(insights.NAMES[c], float(hist[c].min()), float(hist[c].max()), step=100.0, key=f"{k}_{c}")
                for c in LAGS}
        st.caption("Limited to the lowest and highest values seen so far.")
    st.button("Reset to real conditions", on_click=reset)

scenario = dict(real, Academic_Period=period, Price_per_meal=price, **lags)
real_fc, scen_fc = model.forecast(real), model.forecast(scenario)
with right:
    c1, c2, c3 = st.columns(3)
    with c1:
        ui.stat("Real conditions", ui.n(real_fc), "meals")
    with c2:
        ui.stat("Your scenario", ui.n(scen_fc), "meals")
    with c3:
        ui.stat("Difference", f"{scen_fc - real_fc:+,.0f}", f"meals ({(scen_fc - real_fc) / real_fc:+.1%})")
    with st.container(border=True):
        st.subheader("What changed", anchor=False)
        _, rc = explain(tuple(real.items()))
        _, sc = explain(tuple(scenario.items()))
        data = pd.concat([pd.DataFrame({"Factor": [insights.NAMES[v] for v in rc.index], "Meals": rc.values, "Case": "Real conditions"}),
                          pd.DataFrame({"Factor": [insights.NAMES[v] for v in sc.index], "Meals": sc.values, "Case": "Your scenario"})])
        order = [insights.NAMES[v] for v in rc.abs().sort_values(ascending=False).index]
        ui.chart(alt.Chart(data).mark_bar().encode(
            y=alt.Y("Factor:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)), yOffset="Case:N",
            x=alt.X("Meals:Q", title="Meals, compared with a typical day"),
            color=alt.Color("Case:N", scale=alt.Scale(range=[ui.GREY, ui.BLUE]), legend=alt.Legend(orient="bottom", title=None)),
            tooltip=["Case", "Factor", alt.Tooltip("Meals:Q", format="+,.0f")]).properties(height=2 * ui.bar_height(len(rc))), 2 * ui.bar_height(len(rc)) + 80)
    st.info("The meal price changes the forecast only a little, because the demo data was made without a price effect. "
            "The day of the week, month and weekend follow from the chosen day.")
