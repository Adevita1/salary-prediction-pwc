# src/feature_engineering.py

"""
Módulo de ingeniería de características:
- Agrega columnas derivadas útiles para el modelo.
"""

import pandas as pd

def add_features(df):
    """
    Genera nuevas columnas a partir del dataset original.

    Actualmente agrega:
    - 'Title Length': longitud (en caracteres) del campo 'Job Title'.

    Args:
        df (pd.DataFrame): DataFrame original.

    Returns:
        pd.DataFrame: DataFrame con nuevas columnas agregadas.
    """
    df = df.copy()
    df["Title Length"] = df["Job Title"].apply(lambda x: len(str(x)))
    
    return df

