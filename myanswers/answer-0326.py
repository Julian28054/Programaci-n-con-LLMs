import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, componentes: float):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Pasamos la variable con el nuevo nombre al PCA
    pca = PCA(n_components=componentes)
    X_reducido = pca.fit_transform(X_scaled)
    
    return X_reducido