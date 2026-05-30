import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza):
    """
    Reduce la dimensionalidad de un conjunto de datos utilizando PCA,
    conservando al menos el porcentaje de varianza especificado.

    Parámetros:
        X (pd.DataFrame): Datos originales.
        umbral_varianza (float): Valor entre 0 y 1.

    Retorna:
        np.ndarray: Datos transformados al espacio reducido.
    """

    # Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Aplicar PCA conservando la varianza deseada
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)

    return X_reducido
