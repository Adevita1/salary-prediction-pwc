import pandas as pd

def load_data():
    salary_df = pd.read_csv("../data/salary.csv")
    people_df = pd.read_csv("../data/people.csv")
    descriptions_df = pd.read_csv("../data/descriptions.csv")

    df = salary_df.merge(people_df, on="id", how="left")
    df = df.merge(descriptions_df, on="id", how="left")
    return df

def preprocess_input_data(df):
    df = df.copy()

    # Mismo preprocesamiento usado en entrenamiento
    categorical_cols = ['Gender', 'Education Level', 'Job Title']
    for col in categorical_cols:
        df[col] = df[col].astype('category')

    return df
