import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza):
    """
    Reduce la dimensionalidad de un conjunto de datos usando PCA,
    conservando la varianza especificada.

    Parámetros:
    X (pd.DataFrame): Datos originales.
    umbral_varianza (float): Porcentaje de varianza a conservar (0-1).

    Retorna:
    np.ndarray: Datos transformados al espacio reducido.
    """
    
    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA conservando el porcentaje de varianza deseado
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)

    return X_reducido
