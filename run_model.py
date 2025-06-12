import pandas as pd
from src.preprocess import load_and_clean_data
from src.model import train_model
import joblib

df = load_and_clean_data()
model = train_model(df)
joblib.dump(model, 'src/exam_score_model.pkl')
print("✅ Model trained and saved as 'src/exam_score_model.pkl'")
