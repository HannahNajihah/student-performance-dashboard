# run_model.py

from src.preprocess import preprocess_data
from src.model import train_model
import joblib

# Load and preprocess the dataset
df_clean = preprocess_data('data/student_performance.csv')

# Train the model and save it
model = train_model(df_clean)
joblib.dump(model, "src/exam_score_model.pkl")
