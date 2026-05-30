import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # Seleccionar únicamente columnas numéricas
    df_num = df.select_dtypes(include=[np.number])

    # Separar variables predictoras y objetivo
    X = df_num.drop(columns=[target_col])
    y = df_num[target_col]

    # División 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entrenamiento del modelo
    modelo = DecisionTreeRegressor(random_state=42)
    modelo.fit(X_train, y_train)

    # Predicción y cálculo del MAE
    y_pred = modelo.predict(X_test)

    return float(mean_absolute_error(y_test, y_pred))
