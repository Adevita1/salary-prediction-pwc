import os
import pandas as pd

def load_data():
    # Construir ruta absoluta hacia el directorio 'data'
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    # Cargar los CSVs desde la carpeta /data
    salary_df = pd.read_csv(os.path.join(base_path, "salary.csv"))
    people_df = pd.read_csv(os.path.join(base_path, "people.csv"))
    descriptions_df = pd.read_csv(os.path.join(base_path, "descriptions.csv"))

    # Unir los datasets por 'id'
    df = salary_df.merge(people_df, on="id", how="left")
    df = df.merge(descriptions_df, on="id", how="left")

    return df

def preprocess_input_data(df):
    df = df.copy()

    # Convertir columnas categóricas
    categorical_cols = ['Gender', 'Education Level', 'Job Title']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    return df

def load_and_prepare_data():
    df = load_data()
    df = preprocess_input_data(df)
    return df
