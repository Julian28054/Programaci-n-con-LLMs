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
    # 1. Seleccionar X primero (antes de separar y), en el mismo orden que el generador
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 2. Dividir los datos en entrenamiento y prueba (80/20) - SIN random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 3. Entrenar el modelo DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 4. Calcular el error absoluto medio (MAE)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
