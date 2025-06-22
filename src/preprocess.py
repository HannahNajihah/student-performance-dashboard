import pandas as pd
import joblib

def preprocess_data(path='data/student_performance.csv'):
    df = pd.read_csv(path)

    # Remove duplicates and strip whitespace
    df.drop_duplicates(inplace=True)
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Ordinal encoding
    ordinal_map = {
        'Parental_Involvement': ['Low', 'Medium', 'High'],
        'Motivation_Level': ['Low', 'Medium', 'High'],
        'Family_Income': ['Low', 'Medium', 'High']
    }
    for col, levels in ordinal_map.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=levels, ordered=True).codes

    # Binary encoding
    binary_map = {
        'Internet_Access': {'No': 0, 'Yes': 1},
        'Extracurricular_Activities': {'No': 0, 'Yes': 1}
    }
    for col, mapping in binary_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # Save original gender for dashboard filter
    df['Original_Gender'] = df['Gender']
    if 'Gender' in df.columns:
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # One-hot encode any other categorical variables
    remaining_cats = df.select_dtypes(include='object').columns
    if len(remaining_cats) > 0:
        df = pd.get_dummies(df, columns=remaining_cats, drop_first=True)

    # Remove outliers (except Exam_Score) using IQR
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        if col == 'Exam_Score':
            continue
        Q1, Q3 = df[col].quantile([0.25, 0.75])
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    return df
