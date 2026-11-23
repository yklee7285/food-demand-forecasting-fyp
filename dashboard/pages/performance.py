"""Performance: accuracy, training time, explanations and limitations, from the notebook's final run."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import wilcoxon

import insights
import model
import ui

m = model.meta()
tp = model.csv("test_predictions")
res = model.table("results")
s1 = res.loc[[i for i in res.index if i.startswith("Stacking 1")][0]]
naive = res.loc[[i for i in res.index if i.startswith("Seasonal naive")][0]]
tune = model.table("tuning_comparison").loc["Stacking 1"]

ui.header("Performance", "How good is the model?",
          f"Test period {m['test_dates'][0]} to {m['test_dates'][1]}: {len(tp)} days the model never saw in training. "
          "All figures come from the final run of notebooks/synthetic_canteen/canteen_methodology_improved.ipynb.")

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.stat("Average error (MAE)", ui.n(s1["MAE"]), "meals per day")
with c2:
    ui.stat("R²", f"{s1['R2']:.3f}", "the noise in the data caps it at about 0.88")
with c3:
    ui.stat("Better than the baseline by", f"{1 - s1['MAE'] / naive['MAE']:.0%}", f"same weekday last week, MAE {ui.n(naive['MAE'])}")
with c4:
    ui.stat("Training", f"{tune['Total time (s)'].iloc[0] / tune['Total time (s)'].iloc[1]:.1f}× faster", "than the original tuning")

acc, speed, drivers, about = st.tabs(["Accuracy", "Training time", "What drives the forecast", "About and limitations"])

with acc:
    cols = ["MAE", "RMSE", "SMAPE (%)", "R2", "Training time (s)", "Accuracy (%)"]
    show = res[cols].rename(columns={"R2": "R²", "Accuracy (%)": "Accuracy = 100 − MAPE (%)"})
    st.dataframe(show.style.format("{:,.2f}").format("{:.3f}", subset=["R²"]).apply(
        lambda r: ["font-weight:bold;background-color:rgba(29,78,216,.12)" if r.name.startswith("Stacking 1") else "" for _ in r],
        axis=1), width="stretch")
    st.caption("The dashboard uses Stacking 1 (highlighted). Errors are in meals. Accuracy = 100 − MAPE is the paper's metric, "
               "shown for comparison only. Stacking training times cover the stack only.")
    singles = [c for c in ["RF", "XGBoost", "SVR", "LSTM"] if c in tp]
    err = {c: np.abs(tp["Actual"] - tp[c]) for c in singles + ["Stacking 1"]}
    best = min(singles, key=lambda c: err[c].mean())
    p = wilcoxon(err["Stacking 1"], err[best]).pvalue
    st.write(f"Stacking 1 vs the best single model ({best}): MAE {err['Stacking 1'].mean():,.1f} vs {err[best].mean():,.1f}; "
             f"Wilcoxon signed-rank p = {p:.2f}, so the difference is {'' if p < 0.05 else 'not '}significant at the 5 % level.")
    chart = tp[["Actual", "Stacking 1", "Seasonal naive"]].rename(columns={
        "Actual": "Meals sold", "Stacking 1": "Forecast", "Seasonal naive": "Baseline (same weekday last week)"}).rename_axis("Date").reset_index()
    ui.chart(ui.sold_vs_forecast(chart, extra="Baseline (same weekday last week)"), 400)
    st.markdown("**Preparation plan on the test days** (safety margin from the out-of-fold errors on the training days)")
    level = st.session_state.get("level", 80) / 100
    bt = insights.backtest(level)
    st.dataframe(bt.style.format("{:,.0f}"), width="stretch")
    st.caption(f"Days with enough food under the plan: {bt.iloc[0]['Days with enough food'] / len(tp):.0%} (aim {level:.0%}).")

with speed:
    t = tune.reset_index()[["Meta-learner tuning", "alpha", "MAE", "R2", "Stacking time (s)", "Total time (s)"]].rename(columns={"R2": "R²"})
    st.dataframe(t.style.format({"alpha": "{:g}", "MAE": "{:,.1f}", "R²": "{:.3f}", "Stacking time (s)": "{:.1f}",
                                 "Total time (s)": "{:.1f}"}), hide_index=True, width="stretch")
    st.caption("RidgeCV chooses the Ridge alpha from one fit of the stack instead of refitting it for every candidate alpha. "
               "Total time includes the base models' grid searches. Times are single measurements.")

with drivers:
    imp = model.table("global_importance")
    g = pd.DataFrame({"Input": [insights.NAMES.get(v, v) for v in imp.index],
                      "Meals": imp[("Mean |SHAP| (meals)", "Stacking 1")].values, "Share": imp[("Share (%)", "Stacking 1")].values})
    g = g[g.Meals > 0].sort_values("Meals", ascending=False)
    g["Label"] = [f"{v:.1f} %" for v in g.Share]
    bars = alt.Chart(g).mark_bar(color=ui.BLUE).encode(
        x=alt.X("Meals:Q", title="Mean absolute SHAP value (meals)"), y=alt.Y("Input:N", sort=list(g.Input), title=None,
                                                                           axis=alt.Axis(labelLimit=260)),
        tooltip=["Input", alt.Tooltip("Meals:Q", format=",.1f"), alt.Tooltip("Share:Q", format=".1f")])
    ui.chart((bars + bars.mark_text(align="left", dx=3, color="#333").encode(text="Label:N")).properties(
        height=ui.bar_height(len(g))), ui.bar_height(len(g)) + 80)
    st.caption("Global SHAP importance over the test days. Weekend status and day of the week carry the same Saturday "
               "information and should be read together.")
    st.markdown("**LIME for one test day**")
    lday = st.date_input("Test day", value=tp.index[0].date(), min_value=tp.index[0].date(), max_value=tp.index[-1].date(),
                         format="DD/MM/YYYY", key="lime_day")
    if pd.Timestamp(lday).dayofweek == 6:
        st.warning("Sunday is not an operating day.")
    elif st.button("Run LIME", key="lime"):
        hist = model.history()
        cond = model.conditions(lday, hist.loc[hist.index < pd.Timestamp(lday), m["target"]].astype(float))
        with st.spinner("Running LIME..."):
            st.dataframe(model.lime_table(cond).round(1), hide_index=True, width="stretch")
        st.caption("LIME changes each one-hot column on its own, so its conditions use the scaled inputs and only its top "
                   "variables are compared with SHAP's.")

with about:
    st.markdown(f"""
**Project.** Final-year project *Food Waste Prediction System for Campus Canteens*: forecast daily meal demand so that a
canteen prepares closer to what it sells and wastes less food.

**Data.** Synthetic daily meals sold, Monday to Saturday, {m['train_dates'][0]} to {m['test_dates'][1]}, generated from
assumed day-of-week and academic-period effects plus random variation. Not recorded sales.

**Model.** Stacking 1: Random Forest, XGBoost and SVR, combined by a Ridge regression (alpha chosen by RidgeCV). Trained on
days up to {m['train_dates'][1]}; evaluated on the days after. The dashboard loads the model saved by the notebook and never
retrains it. Recorded days only update the recent-sales inputs.

**Explanations.** The reason on the Today page comes from SHAP values (permutation explainer, 100 reference training days):
each forecast is split into a typical day plus one contribution per input. LIME is available above as a second method.

**Limitations.**
- The data is synthetic, so accuracy figures do not describe a real canteen.
- Accuracy is close to the limit set by the random noise in the data (R² about 0.88); all models score about the same.
- Forecasts are one day ahead; the week-ahead view feeds forecasts back in and is a rough guide only.
- The preparation plan assumes future forecast errors look like the training ones.
- Price has no effect in the generated data. Days after 31 December 2026 need the academic period entered by hand.
""")
    v = m["versions"]
    st.caption(f"Model exported {m['exported']} · Python {v['python']} · scikit-learn {v['scikit-learn']} · XGBoost {v['xgboost']} "
               f"· SHAP {v['shap']}")
