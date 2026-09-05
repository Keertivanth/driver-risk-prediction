def preprocess_data(df):
    df = df.copy()

    # clean column names
    df.columns = df.columns.str.strip()

    # convert Class safely
    if 'Class' in df.columns:
        df['Class'] = df['Class'].astype(str).str.upper().map({
            'AGGRESSIVE': 1,
            'NORMAL': 0
        })

    # drop timestamp if exists
    if 'Timestamp' in df.columns:
        df = df.drop(columns=['Timestamp'])

    # fill missing
    df = df.fillna(0)

    return df