import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    """
    Evalúa un modelo DecisionTreeRegressor sobre datos de pavimentos.
    
    Parámetros:
    df : pd.DataFrame
        DataFrame con variables de entrada y la columna objetivo.
    target_col : str
        Nombre de la columna objetivo (ej. 'deflection').
    
    Retorna:
    float : MAE en el conjunto de prueba.
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 2. Dividir en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 3. Entrenar modelo DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Predecir y calcular MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
