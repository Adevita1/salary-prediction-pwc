# src/model_training.py

"""
Módulo de entrenamiento del modelo:
- Prepara las features y variable target.
- Entrena un modelo de regresión (Random Forest).
- Evalúa con métricas básicas y guarda el modelo entrenado.
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def preprocess_features(df):
    """
    Preprocesa el DataFrame para separar X e y.

    Pasos:
    - Elimina filas con Salary nulo.
    - Codifica variables categóricas con one-hot encoding.
    - Separa X (features) de y (target: salario).

    Args:
        df (pd.DataFrame): Dataset limpio con features y target.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: X, y
    """
    df = df.dropna(subset=["Salary"])  # Eliminar registros sin target
    df = pd.get_dummies(df, drop_first=True)  # One-hot encoding
    X = df.drop("Salary", axis=1)
    y = df["Salary"]
    return X, y

def train_model(df):
    """
    Entrena un modelo RandomForestRegressor sobre los datos.

    Args:
        df (pd.DataFrame): DataFrame de entrenamiento.

    Returns:
        model: Modelo entrenado.
        X_test: Features de test.
        y_test: Target de test.
    """
    # Separar datos en features (X) y target (y)
    X, y = preprocess_features(df)

    # División train-test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenar modelo Random Forest
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predicción sobre test set
    y_pred = model.predict(X_test)

    # Métricas básicas
    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    # Guardar modelo entrenado
    joblib.dump(model, "model.pkl")
    print("✅ Modelo guardado como model.pkl")

    return model, X_test, y_test
