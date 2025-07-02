import joblib

def save_model(model, path="models/best_model.pkl"):
    """Guarda el modelo entrenado en formato .pkl"""
    joblib.dump(model, path)

def load_model(path="models/best_model.pkl"):
    """Carga un modelo previamente guardado"""
    return joblib.load(path)
