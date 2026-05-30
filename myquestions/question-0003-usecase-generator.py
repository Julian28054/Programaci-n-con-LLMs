import numpy as np
import pandas as pd

def generar_caso_de_uso_calcular_mse_filtrado():
    n_filas = np.random.randint(40, 80)
    
    reales = np.random.uniform(-5.0, 50.0, n_filas)
    # Forzar negativos o ceros al inicio
    reales[0] = 0.0
    reales[1] = -5.0
    
    predicciones = reales + np.random.normal(0, 3, n_filas)
    df = pd.DataFrame({'reales': reales, 'predicciones': predicciones})
    
    # Cálculo de la solución esperada aplicando el filtro
    df_filtrado = df[df['reales'] > 0]
    errores_cuadrados = (df_filtrado['reales'].values - df_filtrado['predicciones'].values) ** 2
    esperado = float(np.mean(errores_cuadrados))
    
    argumentos = {
        'df': df
    }
    
    return argumentos, esperado
