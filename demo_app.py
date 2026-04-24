import streamlit as st
import numpy as np
import pandas as pd
import time
st.markdown(
    """
    <h1 style='text-align: center; font-size: 48px;'>
    🚨 Can You Break This AI Model?
    </h1>
    """,
    unsafe_allow_html=True
)
st.markdown(f"<h1 style='text-align: center;'>{status}</h1>", unsafe_allow_html=True)
st.set_page_config(layout="wide")

# -------------------------
# TITLE
# -------------------------
st.title("🚨 Can You Break This AI Model?")

# -------------------------
# USER CONTROL
# -------------------------
drift_level = st.slider("Increase Data Drift", 0.0, 1.0, 0.0, 0.05)

inject_bias = st.button("Inject Bias")

# -------------------------
# SIMULATION
# -------------------------
drift = drift_level
fairness = 0.0

if inject_bias:
    fairness = np.random.uniform(0.2, 0.5)

# Stability calculation
stability = max(0, 1 - (drift * 0.6 + fairness * 0.4))

# -------------------------
# STATUS LOGIC
# -------------------------
def get_status(value):
    if value > 0.7:
        return "🟢 STABLE"
    elif value > 0.4:
        return "🟡 AT RISK"
    else:
        return "🔴 FAILURE"

status = get_status(stability)

# -------------------------
# DISPLAY (BIG + VISUAL)
# -------------------------
col1, col2, col3 = st.columns(3)

col1.metric("Drift", round(drift, 2))
col2.metric("Fairness Risk", round(fairness, 2))
col3.metric("Stability", round(stability, 2))

st.markdown(f"# {status}")

# -------------------------
# LIVE CHART
# -------------------------
chart_data = pd.DataFrame({
    "Drift": [drift],
    "Fairness": [fairness],
})

st.line_chart(chart_data)
