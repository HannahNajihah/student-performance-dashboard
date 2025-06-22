from src.preprocess import preprocess_data
from sklearn.linear_model import LinearRegression
import joblib

df = preprocess_data()
X = df.drop('Exam_Score', axis=1)
y = df['Exam_Score']
model = LinearRegression().fit(X, y)
joblib.dump(model, 'src/exam_score_model.pkl')
