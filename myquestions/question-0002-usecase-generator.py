import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

def generar_caso_de_uso_predecir_consumo():
    n_filas = np.random.randint(30, 60)
    
    temp = np.random.uniform(15.0, 35.0, n_filas)
    habitantes = np.random.randint(1, 6, n_filas)
    X = pd.DataFrame({'temperatura': temp, 'habitantes': habitantes})
    
    y = 50 + (temp * -1.5) + (habitantes * 20.0) + np.random.normal(0, 2, n_filas)
    
    nueva_instancia = {
        'temperatura': float(np.random.uniform(15.0, 35.0)),
        'habitantes': int(np.random.randint(1, 6))
    }
    
    # Cálculo de la solución
    modelo = LinearRegression()
    modelo.fit(X, y)
    X_nuevo = pd.DataFrame([nueva_instancia])
    esperado = float(modelo.predict(X_nuevo)[0])
    
    argumentos = {
        'X': X,
        'y': y,
        'nueva_instancia': nueva_instancia
    }
    
    return argumentos, esperado
