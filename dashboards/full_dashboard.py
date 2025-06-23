import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from src.preprocess import preprocess_data

# --- Page Config ---
st.set_page_config(page_title="🎓 MVP: Student Performance", layout="wide")
st.title("🎓 MVP: Student Performance Dashboard")

# --- Load data ---
df = preprocess_data()

# --- Load model ---
model = None
model_path = "src/exam_score_model.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.warning(f"⚠️ Could not load model: {e}")

# --- Sidebar filter ---
st.sidebar.header("🔎 Filter")
attendance = st.sidebar.slider("Minimum Attendance (%)", 0, 100, 75)
df = df[df['Attendance'] >= attendance]

# --- Show key student records ---
st.subheader("📋 Sample Student Records")
sample_cols = ['Hours_Studied', 'Sleep_Hours', 'Attendance', 'Exam_Score']
st.dataframe(df[sample_cols].head(10))

# --- Chart 1: Histogram of Exam Scores ---
st.subheader("📊 Distribution of Exam Scores")
fig1 = px.histogram(df, x="Exam_Score", nbins=20, title="Exam Score Distribution")
st.plotly_chart(fig1, use_container_width=True)

# --- Prediction Section ---
st.markdown("---")
st.subheader("🎯 Predict Exam Score")

if model is not None:
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        sleep = col1.slider("Sleep Hours", 0, 12, 7)
        study = col1.slider("Hours Studied", 0, 40, 10)
        attendance_input = col2.slider("Attendance (%)", 0, 100, 80)
        motivation = col2.selectbox("Motivation Level", ['Low', 'Medium', 'High'])
        involvement = col2.selectbox("Parental Involvement", ['Low', 'Medium', 'High'])
        submitted = st.form_submit_button("Predict")

    if submitted:
        input_dict = {
            "Sleep_Hours": sleep,
            "Hours_Studied": study,
            "Attendance": attendance_input,
            "Motivation_Level": ['Low', 'Medium', 'High'].index(motivation),
            "Parental_Involvement": ['Low', 'Medium', 'High'].index(involvement),
            "Distance_from_Home": 5,
            "Family_Income": 1,
            "Internet_Access": 1,
            "Extracurricular_Activities": 1,
        }

        X_columns = df.drop(columns=['Exam_Score']).columns.tolist()
        input_df = pd.DataFrame([input_dict])
        for col in X_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[X_columns]

        pred = model.predict(input_df)[0]
        st.success(f"✅ Predicted Exam Score: **{pred:.2f}**")
else:
    st.warning("⚠️ Prediction unavailable – model not found.")