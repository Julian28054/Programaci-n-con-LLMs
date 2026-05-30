import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza):
    """
    Reduce la dimensionalidad usando PCA basado en un umbral de varianza acumulada.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame con las características originales
    umbral_varianza : float
        Valor entre 0 y 1 que indica cuánta varianza conservar (ej. 0.95 para 95%)
    
    Returns:
    --------
    numpy.ndarray
        Array con las coordenadas de los datos en el espacio reducido
    """
    # Verificar que el umbral sea válido
    if not 0 <= umbral_varianza <= 1:
        raise ValueError("El umbral de varianza debe estar entre 0 y 1")
    
    # Estandarizar los datos (media=0, varianza=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA con el umbral de varianza
    # Cuando n_components es float entre 0 y 1, selecciona automáticamente
    # los componentes que explican esa varianza acumulada
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    # Retornar SOLO el array numpy (no el diccionario)
    return X_reducido