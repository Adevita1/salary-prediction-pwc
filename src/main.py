# main.py

from src.data_preprocessing import load_and_prepare_data
from src.feature_engineering import add_features
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.save_model import save_model
from src.inference import predict_with_model

import pandas as pd

def main():
    print("🚀 Iniciando pipeline de predicción de salarios...\n")

    # 1. Cargar y preparar los datos
    df = load_and_prepare_data()

    # 2. Agregar nuevas features
    df = add_features(df)

    # 3. Entrenar el modelo
    model, X_test, y_test = train_model(df)

    # 4. Evaluar el modelo
    print("\n🔍 Evaluación del modelo:")
    evaluate_model(model, X_test, y_test)

    # 5. Guardar modelo entrenado
    save_model(model, "trained_model.pkl")

    # 6. Inference de prueba con una fila del set de test
    print("\n📦 Inference de prueba:")
    sample_input = X_test.iloc[[0]]
    predict_with_model("trained_model.pkl", sample_input)

if __name__ == "__main__":
    main()
