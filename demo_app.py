import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(layout="wide")

# -------------------------
# SESSION STATE (CRITICAL)
# -------------------------
if "drift" not in st.session_state:
    st.session_state.drift = 0.0

if "fairness" not in st.session_state:
    st.session_state.fairness = 0.0

# -------------------------
# TITLE
# -------------------------
st.markdown(
    "<h1 style='text-align: center; font-size: 42px;'>🚨 Break This AI Model</h1>",
    unsafe_allow_html=True
)

# -------------------------
# BUTTON CONTROLS (MOBILE FRIENDLY)
# -------------------------
col1, col2 = st.columns(2)

if col1.button("⬆️ Add Drift"):
    st.session_state.drift = min(1.0, st.session_state.drift + 0.15)

if col2.button("⚠️ Add Bias"):
    st.session_state.fairness = min(1.0, st.session_state.fairness + np.random.uniform(0.15, 0.3))

# Reset button (important for repeating demo)
if st.button("🔄 Reset"):
    st.session_state.drift = 0.0
    st.session_state.fairness = 0.0

drift = st.session_state.drift
fairness = st.session_state.fairness

# -------------------------
# SYSTEM LOGIC
# -------------------------
stability = max(0, 1 - (drift * 0.6 + fairness * 0.4))

def get_status(value):
    if value > 0.7:
        return "🟢 STABLE"
    elif value > 0.4:
        return "🟡 AT RISK"
    else:
        return "🔴 FAILURE"

status = get_status(stability)
if status == "🔴 FAILURE":
    st.error("⚠️ SYSTEM FAILURE: Model is no longer reliable")
# -------------------------
# DISPLAY (BIG + CLEAN)
# -------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Drift", round(drift, 2))
c2.metric("Bias Risk", round(fairness, 2))
c3.metric("Stability", round(stability, 2))

st.markdown(
    f"<h1 style='text-align: center; font-size: 50px;'>{status}</h1>",
    unsafe_allow_html=True
)

# -------------------------
# CHART HISTORY (THIS MAKES IT FEEL REAL)
# -------------------------
if "history" not in st.session_state:
    st.session_state.history = []

st.session_state.history.append({
    "Drift": drift,
    "Bias": fairness,
    "Stability": stability
})

df = pd.DataFrame(st.session_state.history)

st.line_chart(df)

# -------------------------
# AUTO SCROLL FEEL
# -------------------------
st.caption("👉 Keep clicking to break the system")
