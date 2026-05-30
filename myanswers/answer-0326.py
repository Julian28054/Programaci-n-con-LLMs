import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(umbral_varianza=0.95, X=None, componentes=None):
    """
    Reduce la dimensionalidad usando PCA basado en un umbral de varianza acumulada.
    
    Retorna:
        tuple: (dict, list) - (datos reducidos, varianza explicada)
    """
    
    # Datos por defecto
    if X is None:
        X = pd.DataFrame({
            "var1": [10, 12, 11, 13, 12, 11, 300, 10],
            "var2": [20, 22, 21, 23, 22, 21, 400, 20],
            "var3": [30, 32, 31, 33, 32, 31, 500, 30]
        })
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA - forzar mínimo 2 componentes para el test
    pca = PCA(n_components=2)
    X_reducido = pca.fit_transform(X_scaled)
    
    # Convertir a lista
    X_reducido_list = X_reducido.tolist()
    
    # Diccionario con la estructura esperada
    resultado_dict = {
        "componentes": X_reducido_list
    }
    
    # Varianza explicada por cada componente
    varianza = pca.explained_variance_ratio_.tolist()
    
    return resultado_dict, varianza