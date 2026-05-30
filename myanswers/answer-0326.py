import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(umbral_varianza=0.95, X=None, componentes=None):
    """Documentación"""
    if X is None:
        X = pd.DataFrame({
            "var1": [10, 12, 11, 13, 12, 11, 300, 10],
            "var2": [20, 22, 21, 23, 22, 21, 400, 20],
            "var3": [30, 32, 31, 33, 32, 31, 500, 30]
        })
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    return {"componentes": X_reducido.tolist()}, pca.explained_variance_ratio_.tolist()