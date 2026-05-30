import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # 1. Separar X e y
    y = df[target_col]
    
    # 2. Seleccionar solo columnas numéricas para X
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    
    # 3. Dividir en entrenamiento (80%) y prueba (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 4. Entrenar DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 5. Calcular MAE en el conjunto de prueba
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return float(mae)
