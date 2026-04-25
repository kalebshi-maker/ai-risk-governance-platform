
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

if "history" not in st.session_state:
    st.session_state.history = []

# -------------------------
# TITLE
# -------------------------
st.markdown(
    "<h1 style='text-align: center; font-size: 42px;'>🚨 Break This AI Model</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<h3 style='text-align:center;'>🎯 Goal: Push the system into FAILURE</h3>",
    unsafe_allow_html=True
)

# -------------------------
# BUTTON CONTROLS (MOBILE FRIENDLY)
# -------------------------
col1, col2 = st.columns(2)

if col1.button("⬆️ Add Drift"):
    st.session_state.drift = min(
        1.0,
        st.session_state.drift + np.random.uniform(0.08, 0.15)  # smoother escalation
    )

if col2.button("⚠️ Add Bias"):
    st.session_state.fairness = min(
        1.0,
        st.session_state.fairness + np.random.uniform(0.15, 0.3)
    )

# Reset button (FULL reset)
if st.button("🔄 Reset"):
    st.session_state.drift = 0.0
    st.session_state.fairness = 0.0
    st.session_state.history = []

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

# -------------------------
# PRE-FAILURE TENSION
# -------------------------
if 0.3 < stability <= 0.4:
    st.warning("⚠️ System nearing critical instability...")

# -------------------------
# FAILURE MOMENT (IMPACT)
# -------------------------
if status == "🔴 FAILURE":
    # 1. Inject CSS for a red flashing background and a text glitch animation
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"] { animation: pulse-red 0.5s infinite; }
        @keyframes pulse-red {
            0%, 100% { background-color: rgba(255, 0, 0, 0); }
            50% { background-color: rgba(255, 0, 0, 0.2); }
        }
        .glitch {
            font-size: 50px; font-weight: bold; color: red; text-align: center;
            text-shadow: 0.05em 0 0 #00fffc, -0.03em -0.04em 0 #fc00ff;
            animation: glitch 725ms infinite;
        }
        @keyframes glitch {
            0% { transform: translate(0); }
            20% { transform: translate(-2px, 2px); }
            40% { transform: translate(-2px, -2px); }
            60% { transform: translate(2px, 2px); }
            80% { transform: translate(2px, -2px); }
            100% { transform: translate(0); }
        }
        </style>
        """, unsafe_allow_html=True)

    # 2. Display the animated message
    st.markdown('<p class="glitch">🚨 AI HAS LOST CONTROL 🚨</p>', unsafe_allow_html=True)
    st.error("⚠️ SYSTEM FAILURE: Model is no longer reliable")

    # 3. Audio (Ensure this is indented exactly like the lines above!)
    # Using a high-reliability link from a different source
    audio_url = "https://gfxsounds.com/wp-content/uploads/2022/12/Futuristic-alarm-or-warning-loopable-2.mp3"
    st.audio(audio_url, format="audio/mp3", autoplay=True)
# -------------------------
# SCORE (GAMIFIED)
# -------------------------
score = int((drift**1.2 + fairness**1.2) * 60)
st.markdown(
    f"<h3 style='text-align:center;'>🏆 Score: {score}</h3>",
    unsafe_allow_html=True
)

# -------------------------
# DISPLAY (BIG + CLEAN)
# -------------------------
c1, c2, c3 = st.columns(3)

c1.metric("Drift", round(drift, 2))
c2.metric("Bias Risk", round(fairness, 2))
c3.metric("Stability", round(stability, 2))

# Better progress logic (intuitive)
st.progress(min(1.0, drift + fairness))

st.markdown(
    f"<h1 style='text-align: center; font-size: 50px;'>{status}</h1>",
    unsafe_allow_html=True
)

# -------------------------
# CHART HISTORY (REAL FEEL)
# -------------------------
st.session_state.history.append({
    "Drift": drift,
    "Bias": fairness,
    "Stability": stability
})

# Keep only last 20 points
st.session_state.history = st.session_state.history[-20:]

df = pd.DataFrame(st.session_state.history)

st.line_chart(df)

# -------------------------
# FINAL CONTEXT LINE
# -------------------------
st.caption("💡 This is what happens in real AI systems — silently.")
st.caption("👉 Keep clicking to break the system")
