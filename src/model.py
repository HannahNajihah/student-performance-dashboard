import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
from preprocess import preprocess_data

# Load and preprocess data
df = preprocess_data()

# Split data
X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'src/exam_score_model.pkl')
print("✅ Model trained and saved as src/exam_score_model.pkl")
