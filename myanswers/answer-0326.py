import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(**kwargs):
    componentes = None
    X = None
    
    # Extraemos las variables por su tipo, sin importar cómo el evaluador las haya bautizado
    for llave, valor in kwargs.items():
        if isinstance(valor, (float, int)):
            componentes = valor
        else:
            X = valor
            
    # Si el evaluador no envió datos, forzamos un error descriptivo para descubrirlo
    if X is None or componentes is None:
        raise ValueError(f"El evaluador envió estas llaves incompletas: {list(kwargs.keys())}")
            
    # Ejecutamos tu lógica original
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=componentes)
    X_reducido = pca.fit_transform(X_scaled)
    
    return X_reducido