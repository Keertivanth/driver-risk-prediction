import pandas as pd

def create_features(df):
    df = df.copy()

    # normalize column names
    df.columns = df.columns.str.strip().str.lower()

    expected_cols = ['accx', 'accy', 'accz', 'gyrox', 'gyroy', 'gyroz']

    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0

    for col in expected_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    df[expected_cols] = df[expected_cols].fillna(0)

    if 'class' in df.columns:
        df = df.drop(columns=['class'])

    return df[expected_cols]