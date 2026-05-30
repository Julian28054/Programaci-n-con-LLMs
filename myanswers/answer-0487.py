import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


def evaluar_modelo_pavimento(df: pd.DataFrame, target_col: str) -> float:
    """
    Entrena un DecisionTreeRegressor para predecir la variable objetivo
    indicada y devuelve el Error Absoluto Medio (MAE) en el conjunto de prueba.

    Parámetros
    ----------
    df : pd.DataFrame
        DataFrame con las variables de diseño de pavimentos y la columna objetivo.
    target_col : str
        Nombre de la columna objetivo (ej. 'deflection').

    Retorna
    -------
    float
        MAE calculado sobre el 20 % de datos reservados para prueba.
    """
    # 1. Separar X e y
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # 2. Seleccionar solo columnas numéricas
    X = X.select_dtypes(include=[np.number])

    # 3. Dividir en entrenamiento (80 %) y prueba (20 %)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # 4. Entrenar el modelo
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X_train, y_train)

    # 5. Calcular MAE en el conjunto de prueba
    y_pred = model.predict(X_test)
    return mean_absolute_error(y_test, y_pred)

