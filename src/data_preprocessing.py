# src/data_preprocessing.py

"""
Módulo de preprocesamiento de datos:
- Carga los archivos CSV de datos.
- Realiza un merge por ID.
- Convierte columnas relevantes a tipo categórico.
"""

import os
import pandas as pd

def load_and_prepare_data():
    """
    Carga y prepara el dataset combinando salary, people y descriptions.

    Returns:
        pd.DataFrame: Dataset combinado y preprocesado.
    """
    # Ruta absoluta a la carpeta /data desde este archivo
    base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

    # Cargar los datasets desde CSV
    salary_df = pd.read_csv(os.path.join(base_path, "salary.csv"))
    people_df = pd.read_csv(os.path.join(base_path, "people.csv"))
    descriptions_df = pd.read_csv(os.path.join(base_path, "descriptions.csv"))

    # Hacer merge de los tres datasets usando 'id' como clave
    df = salary_df.merge(people_df, on="id", how="left")
    df = df.merge(descriptions_df, on="id", how="left")

    # Convertir columnas a tipo categórico si aplica
    df["Gender"] = df["Gender"].astype("category")
    df["Education Level"] = df["Education Level"].astype("category")
    df["Job Title"] = df["Job Title"].astype("category")

    return df
