import pandas as pd
import joblib

def preprocess_data(path='data/student_performance.csv'):
    # Load data
    df = pd.read_csv(path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Clean column names and strip whitespace
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Keep original columns for UI filters
    df['Original_Gender'] = df['Gender']
    df['Original_School_Type'] = df['School_Type']
    df['Original_Peer_Influence'] = df['Peer_Influence']

    # Ordinal encoding
    ordinal_features = {
        'Parental_Involvement': ['Low', 'Medium', 'High'],
        'Motivation_Level': ['Low', 'Medium', 'High'],
        'Family_Income': ['Low', 'Medium', 'High'],
        'Access_to_Resources': ['Low', 'Medium', 'High'],
        'Teacher_Quality': ['Low', 'Medium', 'High'],
        'Parental_Education_Level': ['High School', 'College', 'Postgraduate'],
        'Distance_from_Home': ['Near', 'Moderate', 'Far']
    }
    for col, categories in ordinal_features.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes

    # Binary encoding
    binary_features = {
        'Internet_Access': {'No': 0, 'Yes': 1},
        'Extracurricular_Activities': {'No': 0, 'Yes': 1},
        'Learning_Disabilities': {'No': 0, 'Yes': 1}
    }
    for col, mapping in binary_features.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # One-hot encoding for nominal categorical variables
    one_hot_features = ['Gender', 'School_Type', 'Peer_Influence']
    df = pd.get_dummies(df, columns=one_hot_features, drop_first=True)

    return df
