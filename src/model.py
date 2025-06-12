import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(df):
    X = df.drop('Exam_Score', axis=1)
    y = df['Exam_Score']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # ✅ Replace this path with your actual cleaned dataset
    df = pd.read_csv("data/processed_student_data.csv")

    model = train_model(df)

    # ✅ Save the model where your test expects it
    joblib.dump(model, "src/exam_score_model.pkl")
    print("Model trained and saved to src/exam_score_model.pkl")

