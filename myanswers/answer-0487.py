import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    # 1. Separar X e y
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # 2. División 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2
    )

    # 3. Entrenar modelo
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    # 4. Predicción
    y_pred = model.predict(X_test)

    # 5. MAE
    return mean_absolute_error(y_test, y_pred)
