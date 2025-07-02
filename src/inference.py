# src/inference.py

import pandas as pd
import joblib
from src.data_preprocessing import preprocess_input_data
from src.feature_engineering import add_features

def predict_with_model(model_path, input_df=None):
    """
    Carga un modelo entrenado y predice el salario para un input dado.
    Si no se pasa un input_df, se carga desde 'data/example_input.csv'.
    """
    if input_df is None:
        input_df = pd.read_csv("data/example_input.csv")

    # Preprocesar el input
    df = preprocess_input_data(input_df)
    df = add_features(df)

    # Asegurarse de que las columnas coincidan con el modelo
    model = joblib.load(model_path)

    prediction = model.predict(df)
    print(f"💰 Predicción del salario: {prediction[0]:,.2f}")

