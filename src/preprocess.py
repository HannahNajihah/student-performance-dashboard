import pandas as pd
import joblib  # Required for saving the model

def preprocess_data(path='data/student_performance.csv'):
    # Load data
    df = pd.read_csv(path)

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())

    # Strip whitespace from column names and string entries
    df.columns = df.columns.str.strip()
    df = df.apply(lambda col: col.str.strip() if col.dtype == "object" else col)

    # Ordinal encoding
    ordinal_features = {
        'Parental_Involvement': ['Low', 'Medium', 'High'],
        'Motivation_Level': ['Low', 'Medium', 'High'],
        'Family_Income': ['Low', 'Medium', 'High']
    }
    for col, categories in ordinal_features.items():
        if col in df.columns:
            df[col] = pd.Categorical(df[col], categories=categories, ordered=True).codes

    # Binary encoding
    binary_features = {
        'Internet_Access': {'No': 0, 'Yes': 1},
        'Extracurricular_Activities': {'No': 0, 'Yes': 1}
    }
    for col, mapping in binary_features.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    # One-hot encoding for Gender
    df['Original_Gender'] = df['Gender']
    if 'Gender' in df.columns:
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    # One-hot encode remaining categorical features
    remaining_cats = df.select_dtypes(include='object').columns
    if len(remaining_cats) > 0:
        df = pd.get_dummies(df, columns=remaining_cats, drop_first=True)

    # Print data types for inspection
    print(df.dtypes)

    return df
