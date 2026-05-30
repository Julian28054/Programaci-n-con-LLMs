import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    """
    Evalúa un modelo de regresión para predicción de deflexión
    en diseño de pavimentos.

    Parámetros:
    - df: DataFrame con variables predictoras y la variable objetivo.
    - target_col: nombre de la columna objetivo.

    Retorna:
    - MAE (float)
    """

    # Separar variables predictoras y objetivo
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # División entrenamiento/prueba
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2
    )

    # Entrenar modelo
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # Error absoluto medio
    mae = mean_absolute_error(y_test, y_pred)

    return mae
