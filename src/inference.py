# src/inference.py

import joblib

def predict_with_model(model_path, sample_input):
    # Cargar modelo entrenado
    model = joblib.load(model_path)

    # Asegurar que las columnas coincidan con las usadas durante el entrenamiento
    model_features = model.feature_names_in_
    df = sample_input.reindex(columns=model_features, fill_value=0)

    # Inferencia
    prediction = model.predict(df)
    print("💡 Predicción salarial estimada:", prediction[0])

