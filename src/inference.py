# src/inference.py

"""
Módulo de inferencia:
- Carga un modelo entrenado desde archivo.
- Realiza predicciones sobre un input puntual asegurando que las columnas coincidan.
"""

import joblib

def predict_with_model(model_path, sample_input):
    """
    Carga un modelo y realiza una predicción sobre un input puntual.

    Args:
        model_path (str): Ruta al archivo .pkl del modelo entrenado.
        sample_input (pd.DataFrame): Input con las mismas columnas que en entrenamiento.

    Returns:
        None: Imprime la predicción.
    """
    # Cargar el modelo desde archivo .pkl
    model = joblib.load(model_path)

    # Alinear columnas del input con las del modelo
    model_features = model.feature_names_in_
    df = sample_input.reindex(columns=model_features, fill_value=0)

    # Ejecutar predicción
    prediction = model.predict(df)
    print("💡 Predicción salarial estimada:", prediction[0])
