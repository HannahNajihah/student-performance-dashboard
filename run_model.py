# run_model.py

import pandas as pd
from src.preprocess import preprocess_data
from src.model import train_model
import joblib

# Load and preprocess the dataset
df = pd.read_csv("data/student_performance.csv")
df_clean = preprocess_data(df)

# Train the model and save it
model = train_model(df_clean)
joblib.dump(model, "src/exam_score_model.pkl")
