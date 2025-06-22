import pandas as pd

def preprocess_data(path='data/student_performance.csv'):
    df = pd.read_csv(path).drop_duplicates()
    
    for col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0] if df[col].dtype == 'object' else df[col].median())

    df.columns = df.columns.str.strip()
    df = df.apply(lambda x: x.str.strip() if x.dtype == 'object' else x)

    ordinal = ['Parental_Involvement','Motivation_Level','Family_Income','Access_to_Resources','Teacher_Quality','Parental_Education_Level']
    for col in ordinal:
        if col in df: df[col] = pd.Categorical(df[col]).codes

    for col in ['Internet_Access','Extracurricular_Activities','Learning_Disabilities']:
        if col in df: df[col] = df[col].map({'Yes':1,'No':0})

    if 'Peer_Influence' in df:
        df = pd.get_dummies(df, columns=['Peer_Influence'], drop_first=True)

    if 'Gender' in df:
        df['Original_Gender'] = df['Gender']
        df = pd.get_dummies(df, columns=['Gender'], drop_first=True)

    return df
