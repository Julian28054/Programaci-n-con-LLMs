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
    #    Sin random_state para continuar el mismo flujo aleatorio del evaluador
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 4. Entrenar el modelo (sin random_state por la misma razón)
    model = DecisionTreeRegressor()
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

    # Salida de referencia (sin random_state, igual que la solución)
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    output_data = mean_absolute_error(y_test, model.predict(X_test))

    return input_data, output_data


# ──────────────────────────────────────────────
# Verificación: solución reproduce el MAE esperado
# cuando comparte el mismo estado aleatorio
# ──────────────────────────────────────────────

if __name__ == "__main__":
    import copy

    print("=" * 55)
    print("  Verificación de reproducibilidad del MAE")
    print("=" * 55)

    all_match = True
    for seed in [0, 7, 42, 99, 2024]:
        np.random.seed(seed)
        random.seed(seed)

        # --- El evaluador genera el caso ---
        n = random.randint(20, 40)
        n_features = random.randint(3, 6)
        data = np.random.randn(n, n_features)
        cols = [f'feature_{i}' for i in range(n_features)]
        df = pd.DataFrame(data, columns=cols)
        df['deflection'] = np.random.randn(n)

        # Estado S justo antes del split del generador
        state_S = np.random.get_state()
        rstate_S = random.getstate()

        # Generador calcula MAE esperado
        X = df.drop(columns=['deflection']).select_dtypes(include=[np.number])
        y = df['deflection']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        m = DecisionTreeRegressor()
        m.fit(X_train, y_train)
        expected = mean_absolute_error(y_test, m.predict(X_test))

        # Evaluador restaura estado S y llama a la solución
        np.random.set_state(state_S)
        random.setstate(rstate_S)
        obtained = evaluar_modelo_pavimento(df.copy(), 'deflection')

        match = abs(expected - obtained) < 1e-10
        if not match:
            all_match = False
        status = "✓ MATCH" if match else f"✗ DIFF"
        print(f"  seed={seed:4d} | expected={expected:.6f} | obtained={obtained:.6f} | {status}")

    print("=" * 55)
    print(f"  Resultado global: {'✓ Todos coinciden' if all_match else '✗ Hay diferencias'}")
    print("=" * 55)

