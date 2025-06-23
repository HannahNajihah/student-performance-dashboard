import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocess import preprocess_data

# --- Page Config ---
st.set_page_config(page_title="🛌 MVP Dashboard", layout="wide")
st.title("🛌 MVP: Sleep Hours & Exam Score Dashboard")

# --- Load Data ---
df = preprocess_data()

# --- Sidebar Filters ---
min_sleep, max_sleep = st.sidebar.slider("Sleep Hour Range", int(df['Sleep_Hours'].min()), int(df['Sleep_Hours'].max()), (5, 10))
df = df[df['Sleep_Hours'].between(min_sleep, max_sleep)]

# --- Main Layout ---
col1, col2 = st.columns(2)

# 📊 Scatter Plot
with col1:
    st.subheader("📊 Sleep vs Exam Score")
    fig = px.scatter(
        df,
        x="Sleep_Hours",
        y="Exam_Score",
        labels={"Sleep_Hours": "Sleep Hours", "Exam_Score": "Exam Score"},
        title="Sleep Hours vs Exam Score"
    )
    st.plotly_chart(fig, use_container_width=True)

# 📈 Average Score Line Plot
with col2:
    st.subheader("📈 Average Exam Score by Sleep Hour")
    avg_scores = df.groupby("Sleep_Hours")["Exam_Score"].mean().reset_index()
    fig2 = px.line(
        avg_scores,
        x="Sleep_Hours",
        y="Exam_Score",
        markers=True,
        labels={"Sleep_Hours": "Sleep Hours", "Exam_Score": "Average Score"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# --- Data Preview ---
st.markdown("---")
st.subheader("📄 Data Preview")
st.dataframe(df[["Sleep_Hours", "Exam_Score", "Attendance"]].head())
