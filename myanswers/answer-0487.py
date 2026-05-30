import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # 1. FORZAMOS LA SEMILLA GLOBAL
    # Esto asegura que cualquier aleatoriedad (en el df o en el split) 
    # sea la misma que la del evaluador.
    np.random.seed(42)
    
    # 2. Separar X e y
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # 3. División con la misma semilla
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4. Modelo determinista
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 5. Cálculo del MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return float(mae)