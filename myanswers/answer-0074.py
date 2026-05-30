import numpy as np
from sklearn.ensemble import IsolationForest

def detectar_fraude_multivariado(X, contaminacion):
    # 1. Crear el modelo
    modelo = IsolationForest(
        contamination=contaminacion,
        random_state=42
    )
    
    # 2. Ajustar el modelo
    modelo.fit(X)
    
    # 3. Obtener las predicciones
    predicciones = modelo.predict(X)
    
    # 4. Conservar únicamente las filas normales (predicción = 1)
    X_limpia = X[predicciones == 1]
    
    # 5. Devolver la matriz limpia
    return X_limpia
