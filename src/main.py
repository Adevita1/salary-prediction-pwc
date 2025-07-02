# src/main.py

import pandas as pd
from src.data_preprocessing import load_and_prepare_data
from src.feature_engineering import add_features
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.save_model import save_model
from src.inference import predict_with_model

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

    # 6. Inference de prueba con una fila cruda transformada
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

    raw_sample = raw_sample.drop(columns=["Salary"], errors="ignore")
    raw_sample = pd.get_dummies(raw_sample, drop_first=True)

    # Alinear columnas con las del modelo
    aligned_sample = pd.DataFrame(columns=X_test.columns)
    aligned_sample = pd.concat([aligned_sample, raw_sample], ignore_index=True)
    aligned_sample.fillna(0, inplace=True)

    # Predecir
    predict_with_model("trained_model.pkl", aligned_sample)

if __name__ == "__main__":
    main()

