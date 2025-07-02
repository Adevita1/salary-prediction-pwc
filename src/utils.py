import pandas as pd

def clean_dataset(df):
    df = df.dropna()
    df = pd.get_dummies(df, drop_first=True)
    return df
