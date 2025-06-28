import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocess import preprocess_data

# --- Page Config ---
st.set_page_config(page_title="📉 High-Risk Student Identifier", layout="wide")
st.title("📉 Identify High-Risk Students")

# --- Load Data ---
df = preprocess_data()

# --- Sidebar: Risk Thresholds ---
st.sidebar.header("🔧 Define Risk Criteria")
score_threshold = st.sidebar.slider("📉 Maximum Exam Score (Risk Cutoff)", 0, 100, 50)
study_hours_threshold = st.sidebar.slider("⏱ Maximum Study Hours (Risk Cutoff)", 0, 40, 5)
attendance_threshold = st.sidebar.slider("📛 Minimum Attendance (%)", 0, 100, 70)

# --- Filter based on attendance ---
df = df[df['Attendance'] >= attendance_threshold]

# --- Label high-risk students ---
df['Risk'] = df.apply(
    lambda row: 'High Risk' if row['Hours_Studied'] < study_hours_threshold and row['Exam_Score'] < score_threshold else 'Not High Risk',
    axis=1
)

# --- Chart 1: Attendance vs Exam Score ---
st.subheader("📊 Attendance vs Exam Score")
fig1 = px.scatter(
    df,
    x="Attendance",
    y="Exam_Score",
    trendline="ols",
    color_discrete_sequence=["blue"],
    title="Attendance vs Exam Score"
)
st.plotly_chart(fig1, use_container_width=True)

# --- Chart 2: Study Time vs Exam Score with Risk Highlight ---
st.subheader("🚨 Study Time vs Exam Score (Highlighting High-Risk Students)")
fig2 = px.scatter(
    df,
    x="Hours_Studied",
    y="Exam_Score",
    color="Risk",
    color_discrete_map={"High Risk": "red", "Not High Risk": "green"},
    symbol="Risk",
    title="Study Time vs Exam Score (Red = High-Risk Students)"
)
st.plotly_chart(fig2, use_container_width=True)

# --- High-Risk Students Table ---
st.subheader("📋 High-Risk Student Records")

risk_df = df[df['Risk'] == "High Risk"][['Hours_Studied', 'Attendance', 'Exam_Score']]

if risk_df.empty:
    st.success("🎉 Great news! No high-risk students were detected based on the current thresholds.")
else:
    st.dataframe(risk_df.reset_index(drop=True))
    st.markdown(f"✅ **Total High-Risk Students Detected: {len(risk_df)}**")

# --- Footer ---
st.markdown("---")
st.caption("🧠 Tip: You can adjust the sliders on the left to explore different definitions of risk.")
