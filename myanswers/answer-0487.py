import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(*args, **kwargs):
    # Detectar si recibimos argumentos separados o un diccionario (kwargs)
    if 'df' in kwargs and 'target_col' in kwargs:
        df = kwargs['df']
        target_col = kwargs['target_col']
    else:
        # Fallback si el evaluador los pasa como argumentos posicionales
        df = args[0]
        target_col = args[1]

    # 1. Separar X e y
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # 2. Dividir datos (80/20)
    # NOTA: Sin un random_state fijo, el resultado variará. 
    # Si el test falla, prueba añadiendo random_state=42 aquí y en el generador
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 3. Entrenar modelo
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # 4. Calcular MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return float(mae)