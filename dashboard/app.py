"""Canteen meal forecast dashboard. Run with: streamlit run dashboard/app.py"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import insights  # noqa: E402
import model  # noqa: E402
import store  # noqa: E402
import ui  # noqa: E402

st.set_page_config(page_title="Canteen Meal Forecast", page_icon="🍽️", layout="wide")
ui.style()

pages = [st.Page("pages/today.py", title="Today", default=True),
         st.Page("pages/records.py", title="Records"),
         st.Page("pages/what_if.py", title="What-if"),
         st.Page("pages/performance.py", title="Performance")]
nav = st.navigation(pages)

with st.sidebar:
    st.markdown("**Canteen Meal Forecast**  \nCampus canteen · Monday to Saturday")
    st.slider("Chance of having enough food", 50, 95, int(insights.DEFAULT_LEVEL * 100), 5, format="%d%%", key="level",
              help="Higher means fewer days running short, but more leftovers.")
    log = store.load()
    alerts = [a for a in insights.alerts(insights.outcomes(log), st.session_state.level / 100) if a[0] == "warning"]
    last = f"{log['Date'].iloc[-1]:%a %d %b %Y}" if not log.empty else "nothing yet"
    learned = pd.Timestamp(model.meta()["train_dates"][1])
    st.caption(f"**Last recorded:** {last}  \n**Forecasts learned from sales up to:** {learned:%d %b %Y}")
    if alerts:
        st.warning(f"{len(alerts)} alert{'s' if len(alerts) > 1 else ''} on the Records page.")
    st.info("Demo data: the sales are synthetic, generated for this project.")
    st.toggle("Demo mode", key="demo", help="Shows a button on the Today page that fills in the recorded sales, "
              "for stepping through days quickly in a presentation.")

nav.run()
