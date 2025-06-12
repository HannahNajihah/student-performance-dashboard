import joblib

def test_model_loads():
    model = joblib.load('src/exam_score_model.pkl')
    assert model is not None
