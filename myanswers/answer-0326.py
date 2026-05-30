import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(**kwargs):

    X = kwargs.get("X")
    
    if X is None:
        X = kwargs.get("componentes")

    umbral_varianza = kwargs.get("umbral_varianza")

    X = pd.DataFrame(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=umbral_varianza)

    return pca.fit_transform(X_scaled)
