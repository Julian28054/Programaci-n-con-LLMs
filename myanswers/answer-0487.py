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
    # 1. Separar X e y
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 2. Seleccionar solo columnas numéricas
    X = X.select_dtypes(include=[np.number])
    
    # 3. Dividir los datos en entrenamiento y prueba (80/20) - SIN random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 4. Entrenar el modelo DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 5. Calcular el error absoluto medio (MAE)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
