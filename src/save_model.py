# src/save_model.py

"""
Módulo para persistencia del modelo:
- Guarda y carga modelos entrenados utilizando joblib.
"""

import joblib

def save_model(model, path="models/best_model.pkl"):
    """
    Guarda un modelo entrenado en formato .pkl.

    Args:
        model: Objeto del modelo entrenado (por ejemplo, sklearn).
        path (str): Ruta donde guardar el archivo .pkl.
    """
    joblib.dump(model, path)

def load_model(path="models/best_model.pkl"):
    """
    Carga un modelo previamente guardado desde un archivo .pkl.

    Args:
        path (str): Ruta del archivo .pkl.

    Returns:
        model: Modelo cargado listo para usar.
    """
    return joblib.load(path)
