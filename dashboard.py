import streamlit as st
import pandas as pd
import plotly.express as px
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Traffic Dashboard", layout="wide")

st.title(" AI Traffic Monitoring Dashboard")

# -----------------------------
# LOAD DATA
# -----------------------------
csv_path = "output/final_traffic_analysis.csv"

if not os.path.exists(csv_path):
    st.error(" CSV not found. Run main.py first.")
    st.stop()

df = pd.read_csv(csv_path)

# -----------------------------
# KPI CARDS
# -----------------------------
st.subheader(" Key Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Lane 1 Entry", int(df["Lane1_Entry"].sum()))
col2.metric("Lane 2 Entry", int(df["Lane2_Entry"].sum()))
col3.metric("Total Entry", int(df["Entry_Count"].sum()))

# -----------------------------
# ENTRY GRAPH (FINAL CHOICE)
# -----------------------------
st.subheader(" Lane-wise Vehicle Entry (Flow)")

fig = px.bar(
    df,
    x="Video",
    y=["Lane1_Entry", "Lane2_Entry"],
    barmode="group",
    title="Vehicle Entry per Lane"
)

st.plotly_chart(fig, width='stretch')

# -----------------------------
# SIGNAL TIMING GRAPH
# -----------------------------
st.subheader("⏱ Adaptive Signal Timing")

fig2 = px.line(
    df,
    x="Video",
    y=["Lane1_GreenTime", "Lane2_GreenTime"],
    title="Green Time per Lane"
)

st.plotly_chart(fig2, width='stretch')

# -----------------------------
# EVALUATION METRICS
# -----------------------------
st.subheader("📈 Evaluation Metrics")

if "Precision" in df.columns:

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Precision", f"{df['Precision'].mean():.2f}")
    col2.metric("Recall", f"{df['Recall'].mean():.2f}")
    col3.metric("F1 Score", f"{df['F1_Score'].mean():.2f}")
    col4.metric("Accuracy", f"{df['Accuracy'].mean():.2f}")

    fig3 = px.bar(
        df,
        x="Video",
        y=["Precision", "Recall", "F1_Score", "Accuracy"],
        barmode="group",
        title="Evaluation Metrics"
    )

    st.plotly_chart(fig3, width='stretch')

else:
    st.warning(" Metrics not found. Run main.py again.")

# -----------------------------
# HEATMAP DISPLAY
# -----------------------------
st.subheader(" Traffic Heatmaps")

heatmaps = [f for f in os.listdir("output") if "heatmap" in f]

if heatmaps:
    cols = st.columns(2)

    for i, hm in enumerate(heatmaps):
        cols[i % 2].image(f"output/{hm}", caption=hm, use_container_width=True)
else:
    st.warning(" No heatmaps found. Run main.py.")

# -----------------------------
# DATA TABLE
# -----------------------------
st.subheader(" Raw Data")

st.dataframe(df)
