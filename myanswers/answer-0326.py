import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(componentes, umbral_varianza):
    # Convertir a DataFrame por si llega como lista o array
    X = pd.DataFrame(componentes)

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA conservando la varianza solicitada
    pca = PCA(n_components=umbral_varianza)

    # Retornar los datos transformados
    return pca.fit_transform(X_scaled)
