import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# -------------------------------------------------
# DEFINICIÓN DE LA FUNCIÓN SOLICITADA
# -------------------------------------------------
def evaluar_modelo_pavimento(df, target_col):
    """
    Evalúa un modelo de árbol de decisión para predecir la deflexión en pavimentos.
    
    Pasos:
    1. Separa X e y usando target_col.
    2. Selecciona solo columnas numéricas.
    3. Divide los datos en entrenamiento y prueba (80/20).
    4. Entrena un modelo DecisionTreeRegressor.
    5. Calcula el MAE en el conjunto de prueba.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos.
        target_col (str): Nombre de la columna objetivo ('deflection').
        
    Returns:
        float: Valor del Error Absoluto Medio (MAE).
    """
    # 1. Separar X e y
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # 2. Seleccionar solo columnas numéricas
    X = X.select_dtypes(include=[np.number])
    
    # 3. Dividir los datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Entrenar el modelo DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 5. Calcular el error absoluto medio (MAE)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae

# -------------------------------------------------
# CÓDIGO DE PRUEBA (USANDO TU GENERADOR)
# -------------------------------------------------

def generar_caso_de_uso_evaluar_modelo_pavimento():
    """ Genera un caso de prueba para evaluar_modelo_pavimento """
    n = random.randint(20, 40)
    n_features = random.randint(3, 6)
    data = np.random.randn(n, n_features)
    cols = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(data, columns=cols)
    target_col = 'deflection'
    df[target_col] = np.random.randn(n)
    input_data = { 'df': df.copy(), 'target_col': target_col }
    
    # OUTPUT esperado (Cálculo interno para verificación)
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    output_data = mae
    return input_data, output_data

# Ejecución de la prueba
if __name__ == "__main__":
    # 1. Generar el caso de prueba
    input_data, expected_mae = generar_caso_de_uso_evaluar_modelo_pavimento()
    
    # 2. Obtener el DataFrame y la columna objetivo
    df_test = input_data['df']
    target = input_data['target_col']
    
    print(f"DataFrame-shape: {df_test.shape}")
    print(f"Target column: {target}")
    
    # 3. Llamar a la função solicitada
    result_mae = evaluar_modelo_pavimento(df_test, target)
    
    print(f"\nMAE Calculado por la función: {result_mae}")
    print(f"MAE Esperado (referencia):     {expected_mae}")
