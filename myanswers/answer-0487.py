import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # Mantener solo columnas numéricas
    df = df.select_dtypes(include=[np.number])

    # Separar variables
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # División 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Modelo
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Predicción
    y_pred = model.predict(X_test)

    # MAE
    return mean_absolute_error(y_test, y_pred)
