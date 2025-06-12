import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def load_and_clean_data(path='data/student_performance.csv'):
    # Load dataset
    df = pd.read_csv(path)

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # Strip whitespaces in column names and string values
    df.columns = df.columns.str.strip()
    df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

    # Ordinal Encoding
    ordinal_mappings = {
        'Parental_Involvement': ['Low', 'Medium', 'High'],
        'Motivation_Level': ['Low', 'Medium', 'High'],
        'Family_Income': ['Low', 'Medium', 'High']
    }

    for col, order in ordinal_mappings.items():
        df[col] = pd.Categorical(df[col], categories=order, ordered=True).codes

    # Binary Encoding
    binary_mappings = {
        'Internet_Access': {'No': 0, 'Yes': 1},
        'Extracurricular_Activities': {'No': 0, 'Yes': 1}
    }

    for col, mapping in binary_mappings.items():
        df[col] = df[col].map(mapping)

    # One-hot encode categorical features (if any)
    if 'Gender' in df.columns:
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # Remove outliers based on IQR (optional)
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) & (df[col] <= upper)]

    return df


def get_features_and_target(df, target_column='Exam_Score', scale=True):
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    if scale:
        scaler = MinMaxScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
        return X_scaled, y, scaler
    else:
        return X, y, None


def get_train_test_data(df, test_size=0.2, random_state=42):
    X, y, scaler = get_features_and_target(df)
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
