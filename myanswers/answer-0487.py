import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    """
    Evalúa un modelo de árbol de decisión para predecir la deflexión en pavimentos.
    """
    # Eliminar la columna objetivo primero
    X = df.drop(columns=[target_col])
    # Luego seleccionar solo numéricas (mismo orden que el generador)
    X = X.select_dtypes(include=[np.number])
    # Obtener y
    y = df[target_col]
    
    # Dividir sin random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Entrenar
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # Predecir y calcular MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
