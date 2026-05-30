import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(**kwargs):
    # 1. Extraemos el parámetro que ya sabemos que existe
    componentes = kwargs['componentes']
    
    # 2. Extraemos los datos dinámicamente (es la llave que sobra en el diccionario)
    llave_datos = [k for k in kwargs.keys() if k != 'componentes'][0]
    X = kwargs[llave_datos]
    
    # 3. Ejecutamos tu lógica original (que está perfecta)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=componentes)
    X_reducido = pca.fit_transform(X_scaled)
    
    return X_reducido