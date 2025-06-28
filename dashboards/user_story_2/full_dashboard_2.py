import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.preprocess import preprocess_data

# --- Page Config ---
st.set_page_config(page_title="🛌 Sleep & Exam Score Dashboard", layout="wide")
st.title("🛌 Sleep Hours & Exam Score Interactive Dashboard")

# --- Load Data ---
df = preprocess_data()

# --- Load Model ---
model = None
model_path = "src/exam_score_model.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}")

# --- Sidebar Filters ---
st.sidebar.header("🔎 Filter Data")
attendance_filter = st.sidebar.slider("Minimum Attendance (%)", 0, 100, 70)
df = df[df['Attendance'] >= attendance_filter]

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
        color="Original_Gender" if 'Original_Gender' in df.columns else None,
        trendline="ols",
        labels={"Sleep_Hours": "Sleep Hours", "Exam_Score": "Exam Score"},
        title="Scatter Plot with Trendline"
    )
    st.plotly_chart(fig, use_container_width=True)

# 📈 Average Score Line Plot
with col2:
    st.subheader("📈 Average Exam Score per Sleep Hour")
    avg_scores = df.groupby('Sleep_Hours')['Exam_Score'].mean().reset_index()
    fig2 = px.line(avg_scores, x='Sleep_Hours', y='Exam_Score', markers=True,
                   labels={"Sleep_Hours": "Sleep Hours", "Exam_Score": "Average Score"})
    st.plotly_chart(fig2, use_container_width=True)



# --- Prediction Section ---
st.markdown("---")
st.subheader("🎯 Predict Exam Score")

if model is not None:
    with st.form("prediction_form"):
        colA, colB, colC = st.columns(3)

        sleep = colA.slider("Sleep Hours", 0, 12, 7)
        study = colA.slider("Hours Studied", 0, 40, 10)
        attendance = colB.slider("Attendance (%)", 0, 100, 80)
        submitted = st.form_submit_button("Predict")

    if submitted:
        # Prepare input
        input_dict = {
            "Sleep_Hours": sleep,
            "Hours_Studied": study,
            "Attendance": attendance,
            "Distance_from_Home": 5,
            "Family_Income": 1,
            "Internet_Access": 1,
            "Extracurricular_Activities": 1,
        }

      
        # Get model features
        X_columns = df.drop(columns=['Exam_Score']).columns.tolist()

        # Fill missing features
        input_df = pd.DataFrame([input_dict])
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_columns]

        # Predict
        pred = model.predict(input_df)[0]
        st.success(f"✅ Predicted Exam Score: **{pred:.2f}**")
else:
    st.warning("⚠️ Prediction disabled: Model not found.")
