# src/main.py

"""
Este es el script principal del pipeline de predicción de salarios.
Ejecuta los pasos de carga, procesamiento, entrenamiento, evaluación, guardado
y prueba de inferencia con un modelo de regresión.
"""

import pandas as pd
from src.data_preprocessing import load_and_prepare_data
from src.feature_engineering import add_features
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.save_model import save_model
from src.inference import predict_with_model

def main():
    print("🚀 Iniciando pipeline de predicción de salarios...\n")

    # 1. Cargar y preparar los datos (merge + preprocesamiento)
    df = load_and_prepare_data()

    # 2. Generar nuevas features (ejemplo: longitud del título)
    df = add_features(df)

    # 3. Entrenar el modelo de regresión
    model, X_test, y_test = train_model(df)

    # 4. Evaluar el rendimiento del modelo en test set
    print("\n🔍 Evaluación del modelo:")
    evaluate_model(model, X_test, y_test)

    # 5. Guardar el modelo entrenado a disco
    save_model(model, "trained_model.pkl")

    # 6. Simular una predicción con un registro nuevo
    print("\n📦 Inference de prueba:")
    raw_sample = pd.DataFrame([{
        "id": 999,
        "Gender": "Female",
        "Education Level": "Master",
        "Job Title": "Machine Learning Engineer",
        "Description": "Responsible for designing and deploying ML models.",
        "Salary": 0
    }])

    print("\n🔍 Antes de add_features:")
    print(raw_sample)

    # Aplicar las mismas transformaciones que en entrenamiento
    raw_sample = add_features(raw_sample)
    print("\n🔍 Después de add_features:")
    print(raw_sample)

    # Quitar columna de salario si está presente
    raw_sample = raw_sample.drop(columns=["Salary"], errors="ignore")

    # Aplicar one-hot encoding
    raw_sample = pd.get_dummies(raw_sample, drop_first=True)

    # Alinear columnas con las del modelo (agrega columnas faltantes con 0)
    aligned_sample = pd.DataFrame(columns=X_test.columns)
    aligned_sample = pd.concat([aligned_sample, raw_sample], ignore_index=True)
    aligned_sample.fillna(0, inplace=True)

    # Ejecutar predicción final
    predict_with_model("trained_model.pkl", aligned_sample)

if __name__ == "__main__":
    main()
