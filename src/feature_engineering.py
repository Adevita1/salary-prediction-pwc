import os
import pandas as pd

def load_data():
    # Ruta base del directorio "data" desde este archivo .py
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    # Cargamos los CSV usando rutas completas
    salary_df = pd.read_csv(os.path.join(base_path, "salary.csv"))
    people_df = pd.read_csv(os.path.join(base_path, "people.csv"))
    descriptions_df = pd.read_csv(os.path.join(base_path, "descriptions.csv"))

    # Merge de los datasets
    df = salary_df.merge(people_df, on="id", how="left")
    df = df.merge(descriptions_df, on="id", how="left")

    return df

def add_features(df):
    # Ejemplo básico: agregar una columna de longitud del título
    df = df.copy()
    df["Title Length"] = df["Job Title"].apply(lambda x: len(str(x)))
    
    # Podés agregar más features aquí
    return df




