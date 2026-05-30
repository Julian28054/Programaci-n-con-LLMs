import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # Seleccionar solo columnas numéricas
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # Dividir en entrenamiento y prueba (80/20) sin random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Entrenar modelo DecisionTreeRegressor sin random_state
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Predecir y calcular MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    # Devolver el MAE tal cual, sin redondear ni transformar
    return mae
