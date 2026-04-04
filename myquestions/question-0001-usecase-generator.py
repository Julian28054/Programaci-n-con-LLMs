import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def generar_caso_de_uso_limpiar_y_escalar():
    # 1. Crear datos aleatorios con un tamaño variable
    n_filas = np.random.randint(50, 100)
    col_name = "voltaje"
    
    # Generamos datos normales y le sumamos un outlier masivo al principio
    datos = np.random.uniform(10.0, 50.0, n_filas)
    datos[0] = 500.0  # Outlier garantizado
    
    df = pd.DataFrame({col_name: datos})
    
    # 2. Calcular la solución esperada de forma determinista
    df_copia = df.copy()
    p99 = np.percentile(df_copia[col_name], 99)
    
    # Truncar al percentil 99
    df_copia.loc[df_copia[col_name] > p99, col_name] = p99
    
    # Escalar entre 0 y 1
    scaler = MinMaxScaler()
    esperado = scaler.fit_transform(df_copia[[col_name]]).flatten()
    
    # 3. Empaquetar argumentos y retornar
    argumentos = {
        'df': df,
        'columna': col_name
    }
    
    return argumentos, esperado
