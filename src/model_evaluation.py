# src/model_evaluation.py

"""
Módulo para evaluar el desempeño del modelo entrenado.
Incluye métricas como MSE y R² sobre el conjunto de test.
"""

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    """
    Evalúa el modelo usando métricas comunes de regresión.

    Args:
        model: Modelo previamente entrenado.
        X_test (pd.DataFrame): Conjunto de features para test.
        y_test (pd.Series): Valores reales del target.

    Returns:
        tuple: (mse, r2) valores de las métricas obtenidas.
    """
    # Generar predicciones
    y_pred = model.predict(X_test)

    # Calcular métricas
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Mostrar resultados
    print(f"📊 Mean Squared Error: {mse:,.2f}")
    print(f"📈 R² Score: {r2:.2f}")

    return mse, r2
