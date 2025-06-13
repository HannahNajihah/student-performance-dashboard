import streamlit as st
import pandas as pd
from src.preprocess import preprocess_data

st.title("🎓 MVP: Student Performance Dashboard")

# Load and clean dataset
df = preprocess_data()

# Gender filter (if column exists)
if 'Original_Gender' in df.columns:
    selected_gender = st.selectbox("Select Gender", df['Original_Gender'].dropna().unique())
    df = df[df['Original_Gender'] == selected_gender]

# Attendance filter
attendance = st.slider("Minimum Attendance (%)", 0, 100, 75)
filtered_df = df[df['Attendance'] >= attendance]

# Display filtered data
st.subheader("📋 Filtered Student Records")
st.dataframe(filtered_df)

# Show basic statistics
st.subheader("📊 Summary Statistics")
st.write(filtered_df.describe())
