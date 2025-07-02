import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import joblib

def preprocess_features(df):
    df = df.dropna(subset=["Salary"])  # Remove rows without target
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("Salary", axis=1)
    y = df["Salary"]
    return X, y

def train_model(df):
    X, y = preprocess_features(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("R² Score:", r2_score(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))

    joblib.dump(model, "model.pkl")
    print("✅ Modelo guardado como model.pkl")

    return model, X_test, y_test
