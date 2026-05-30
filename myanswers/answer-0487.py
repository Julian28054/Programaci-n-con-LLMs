import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(*args, **kwargs):
    # Obtener df y target_col de forma flexible
    if 'df' in kwargs and 'target_col' in kwargs:
        df = kwargs['df']
        target_col = kwargs['target_col']
    else:
        df = args[0]
        target_col = args[1]

    # 1. Separar X e y
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # 2. Dividir datos con SEMILLA FIJA (Esto es lo que hará que los resultados coincidan)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Entrenar modelo con SEMILLA FIJA
    # También fijamos el random_state en el árbol para evitar variaciones internas
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 4. Calcular MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return float(mae)