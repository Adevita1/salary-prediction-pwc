# src/utils.py

"""
Módulo de utilidades generales del proyecto.
Incluye funciones reutilizables como limpieza y codificación del dataset.
"""

import pandas as pd

def clean_dataset(df):
    """
    Limpia un DataFrame eliminando valores nulos y realizando one-hot encoding.

    Pasos:
    - Elimina cualquier fila con valores faltantes.
    - Aplica codificación one-hot a variables categóricas (omitiendo la primera categoría).

    Args:
        df (pd.DataFrame): Dataset original con datos sin limpiar.

    Returns:
        pd.DataFrame: Dataset limpio y codificado.
    """
    df = df.dropna()  # Eliminar filas con NaN
    df = pd.get_dummies(df, drop_first=True)  # Codificación categórica
    return df
