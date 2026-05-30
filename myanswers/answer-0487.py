import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # Seleccionar columnas numéricas
    try:
        X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    except Exception:
        return None
    y = df[target_col]

    # División 80/20 sin random_state
    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    except Exception:
        return None

    # Entrenar modelo
    try:
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    except Exception:
        return None

    # Calcular MAE y devolverlo tal cual
    try:
        mae = mean_absolute_error(y_test, y_pred)
    except Exception:
        return None

    # No redondear, no transformar, devolver directo
    return mae

