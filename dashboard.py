import joblib
import os
import streamlit as st
import pandas as pd
from src.preprocess import load_and_clean_data

# Load model
model_path = "src/exam_score_model.pkl"
if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        model = None
else:
    st.error("❌ Model file not found. Please train and save the model first.")
    model = None

# Load data
df = load_and_clean_data()

st.title("🎓 Student Performance Dashboard")

# Gender selection
selected_gender = None
gender_male_flag = None

if 'Original_Gender' in df.columns:
    gender_options = df['Original_Gender'].dropna().unique()
    selected_gender = st.selectbox("Select Gender", gender_options)
    df = df[df['Original_Gender'] == selected_gender]

    # Define gender flag if needed
    if 'Gender_Male' in df.columns:
        gender_male_flag = 1 if selected_gender == 'Male' else 0

# Attendance slider
attendance = st.slider("Minimum Attendance (%)", 0, 100, 75)
filtered_df = df[df['Attendance'] >= attendance]

st.write("📋 Filtered Students", filtered_df)

# Prediction section
st.subheader("📈 Predict Exam Score")

hours_studied = st.number_input("Hours Studied", 0, 40)
sleep_hours = st.number_input("Sleep Hours", 0, 12)

# Build base sample
sample_input = {
    'Hours_Studied': hours_studied,
    'Sleep_Hours': sleep_hours,
    'Attendance': attendance,
    'Distance_from_Home': 5,
    'Parental_Involvement': 1,
    'Motivation_Level': 1,
    'Family_Income': 1,
    'Internet_Access': 1,
    'Extracurricular_Activities': 1,
}

# Add gender column if needed
if 'Gender_Male' in load_and_clean_data().columns and gender_male_flag is not None:
    sample_input['Gender_Male'] = gender_male_flag

# Get training features
X_columns = load_and_clean_data().drop(columns=['Exam_Score']).columns.tolist()

# Build a DataFrame for prediction
sample_df = pd.DataFrame([sample_input])

# Add missing columns with default value 0
for col in X_columns:
    if col not in sample_df.columns:
        sample_df[col] = 0

# Reorder columns to match model input
sample_df = sample_df[X_columns]

# Predict
if model is not None:
    try:
        pred = model.predict(sample_df)[0]
        st.success(f"Predicted Exam Score: {pred:.2f}")
        st.metric("Predicted Exam Score", f"{pred:.2f}")
    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
else:
    st.warning("⚠️ Prediction skipped because model is not loaded.")

