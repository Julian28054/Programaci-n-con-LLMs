import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

def generar_caso_de_uso_limpiar_y_escalar():
    """
    Genera un caso de uso para el reto 'Limpieza de Outliers y Escalamiento'.
    Retorna una tupla: (dict_argumentos, objeto_resultado_esperado)
    """
    # --- 1. Creación de Datos Aleatorios ---
    np.random.seed(42)
    n_rows = 30
    col_name = "voltaje"
    
    # Generamos datos normales y forzamos un outlier masivo al inicio
    datos = np.random.uniform(10.0, 50.0, n_rows)
    datos[0] = 500.0  
    
    df_input = pd.DataFrame({col_name: datos})
    # Añadimos una columna objetivo dummy para poder entrenar un Ridge y sacar el R2
    df_input['target'] = df_input[col_name] * 0.5 + np.random.normal(0, 1, n_rows)

    # --- 2. Lógica de Referencia ---
    df = df_input.copy()
    
    # Truncar al percentil 99
    p99 = np.percentile(df[col_name], 99)
    df.loc[df[col_name] > p99, col_name] = p99
    
    # Escalar entre 0 y 1
    scaler = MinMaxScaler()
    df[col_name] = scaler.fit_transform(df[[col_name]])
    
    # Separar y entrenar para obtener métricas tipo el código de tu amigo
    X = df[[col_name]]
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    modelo = Ridge()
    modelo.fit(X_train, y_train)
    r2 = modelo.score(X_test, y_test)

    # --- 3. Construcción de la Tupla Final ---
    argumentos = {
        "df": df_input,
        "columna": col_name
    }
    
    resultado_esperado = {
        "r2_score": r2,
        "model_type": type(modelo),
        "scaler_type": type(scaler)
    }

    return (argumentos, resultado_esperado)


# --- EJEMPLO DE USO ---
inputs, targets = generar_caso_de_uso_limpiar_y_escalar()

print("--- ARGUMENTOS DE ENTRADA (Dataset) ---")
print(inputs['df'].head())
print("\nColumna a evaluar:", inputs['columna'])

print("\n" + "="*40 + "\n")

print("--- RESULTADO ESPERADO ---")
print(f"R2 Score esperado: {targets['r2_score']:.4f}")
print(f"Tipo de modelo esperado: {targets['model_type']}")
print(f"Tipo de scaler esperado: {targets['scaler_type']}")
