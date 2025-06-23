# dashboards/user_story_3/full_dashboard.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
from src.preprocess import preprocess_data

st.title("📘 Motivation Filter Dashboard")

df = preprocess_data()

# Map motivation to readable labels
motivation_map = {0: "Low", 1: "Medium", 2: "High"}
df['Motivation_Label'] = df['Motivation_Level'].map(motivation_map)

# Dropdown to select
selected = st.selectbox("Select Motivation Level", df['Motivation_Label'].unique())

filtered_df = df[df['Motivation_Label'] == selected]

st.subheader(f"🎓 Students with {selected} Motivation")
st.dataframe(filtered_df[['Motivation_Label', 'Exam_Score', 'Hours_Studied', 'Attendance']])

st.subheader("📊 Summary Stats")
st.write(filtered_df.describe())

# --- Chart 2: Line Chart: Avg Exam Score by Hours Studied ---
st.subheader("📈 Average Exam Score by Hours Studied")
avg_by_study = df.groupby('Hours_Studied')['Exam_Score'].mean().reset_index()
median_score = df['Exam_Score'].median()

fig2 = px.line(
    avg_by_study,
    x='Hours_Studied',
    y='Exam_Score',
    markers=True,
    labels={"Hours_Studied": "Hours Studied", "Exam_Score": "Average Score"},
    title="Average Exam Score by Hours Studied"
)

# Add horizontal line for median
fig2.add_shape(
    type="line",
    x0=avg_by_study['Hours_Studied'].min(),
    x1=avg_by_study['Hours_Studied'].max(),
    y0=median_score,
    y1=median_score,
    line=dict(color="Red", width=2, dash="dash"),
)

# Add annotation
fig2.add_annotation(
    x=avg_by_study['Hours_Studied'].median(),
    y=median_score,
    text=f"Median: {median_score:.1f}",
    showarrow=False,
    yshift=10,
    font=dict(color="red")
)

st.plotly_chart(fig2, use_container_width=True)
