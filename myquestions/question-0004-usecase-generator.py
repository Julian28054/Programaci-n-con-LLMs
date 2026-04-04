import numpy as np
import pandas as pd

def generar_caso_de_uso_normalizar_por_grupo():
    n_filas = np.random.randint(60, 100)
    
    sucursales = np.random.choice(['Norte', 'Sur', 'Centro'], n_filas)
    ventas = np.zeros(n_filas)
    
    ventas[sucursales == 'Norte'] = np.random.normal(100, 10, np.sum(sucursales == 'Norte'))
    ventas[sucursales == 'Sur'] = np.random.normal(500, 50, np.sum(sucursales == 'Sur'))
    ventas[sucursales == 'Centro'] = np.random.normal(50, 5, np.sum(sucursales == 'Centro'))
    
    df = pd.DataFrame({'sucursal': sucursales, 'ventas': ventas})
    
    # Cálculo de la solución esperada
    df_copia = df.copy()
    df_copia['ventas_norm'] = df_copia.groupby('sucursal')['ventas'].transform(
        lambda x: (x - x.mean()) / x.std()
    )
    esperado = df_copia['ventas_norm'].values
    
    argumentos = {
        'df': df
    }
    
    return argumentos, esperado
