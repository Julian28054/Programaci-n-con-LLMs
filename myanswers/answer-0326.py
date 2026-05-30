import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza=0.95, componentes=None):
    """
    Reduce la dimensionalidad de un dataset usando PCA basado en un umbral de varianza acumulada.
    
    Argumentos:
    -----------
    X : pd.DataFrame
        DataFrame con las características originales
    umbral_varianza : float
        Valor entre 0 y 1 (ej. 0.95 para 95%)
    componentes : any
        Parámetro ignorado (compatibilidad con generador)
    
    Retorna:
    --------
    numpy.ndarray
        Array con las nuevas coordenadas en el espacio reducido
    """
    
    # Paso 1: Escalar los datos (importante para PCA)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Paso 2: Aplicar PCA con el umbral de varianza
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    # Paso 3: Retornar los datos transformados
    return X_reducido
