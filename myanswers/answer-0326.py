import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X=None, umbral_varianza=0.95, componentes=None):
    """
    Reduce la dimensionalidad usando PCA basado en un umbral de varianza acumulada.
    
    Argumentos:
    -----------
    X : pd.DataFrame
        DataFrame con las características originales
    umbral_varianza : float
        Valor entre 0 y 1 (ej. 0.95 para 95%)
    componentes : any
        Parámetro de compatibilidad (ignorado)
    
    Retorna:
    --------
    tuple: (dict, list)
        - Diccionario con las coordenadas reducidas
        - Lista con la varianza explicada por cada componente
    """
    
    # Si X es None, usar datos de ejemplo del problema
    if X is None:
        X = pd.DataFrame({
            "var1": [10, 12, 11, 13, 12, 11, 300, 10],
            "var2": [20, 22, 21, 23, 22, 21, 400, 20],
            "var3": [30, 32, 31, 33, 32, 31, 500, 30]
        })
    
    # 1. Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Aplicar PCA
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    # 3. Retornar 2 elementos
    resultado_dict = {
        "componentes": X_reducido.tolist()
    }
    
    varianza_explicada = pca.explained_variance_ratio_.tolist()
    
    return resultado_dict, varianza_explicada
