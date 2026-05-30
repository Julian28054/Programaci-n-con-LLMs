import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza: float):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    return X_reducido