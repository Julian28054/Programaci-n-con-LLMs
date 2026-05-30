import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # 1. Separar X e y
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # 2. Partición determinista (NO aleatoria)
    # En lugar de train_test_split, tomamos el 80% inicial para train 
    # y el 20% final para test. Esto es constante y no depende del azar.
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 3. Entrenar modelo
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # 4. Calcular MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return float(mae)