import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error


# ──────────────────────────────────────────────
# Función principal
# ──────────────────────────────────────────────

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
        X, y, test_size=0.2, random_state=42
    )

    # 4. Entrenar el modelo
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # 5. Calcular MAE en el conjunto de prueba
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return mae


# ──────────────────────────────────────────────
# Generador de caso de uso (provisto en el enunciado)
# ──────────────────────────────────────────────

def generar_caso_de_uso_evaluar_modelo_pavimento():
    """
    Genera un caso de prueba para evaluar_modelo_pavimento.
    Retorna (input_data, output_esperado).
    """
    n = random.randint(20, 40)
    n_features = random.randint(3, 6)
    data = np.random.randn(n, n_features)
    cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    target_col = 'deflection'
    df[target_col] = np.random.randn(n)

    input_data = {
        'df': df.copy(),
        'target_col': target_col,
    }

    # Salida de referencia (sin random_state fijo → puede diferir de la función)
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    output_data = mean_absolute_error(y_test, y_pred)

    return input_data, output_data


# ──────────────────────────────────────────────
# Demo de ejecución
# ──────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)

    input_data, mae_referencia = generar_caso_de_uso_evaluar_modelo_pavimento()

    mae_obtenido = evaluar_modelo_pavimento(
        df=input_data['df'],
        target_col=input_data['target_col'],
    )

    print("=" * 45)
    print("  evaluar_modelo_pavimento — demo")
    print("=" * 45)
    print(f"  Filas del dataset   : {len(input_data['df'])}")
    print(f"  Columnas numéricas  : {input_data['df'].select_dtypes(include=[np.number]).shape[1] - 1} features + 1 target")
    print(f"  MAE obtenido        : {mae_obtenido:.6f}")
    print(f"  MAE de referencia   : {mae_referencia:.6f}")
    print("=" * 45)
    print("  Nota: diferencia esperada por distintos random_state.")


