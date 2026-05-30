import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(umbral_varianza=0.95, X=None, componentes=None):
    """
    Reduce la dimensionalidad usando PCA basado en un umbral de varianza acumulada.
    
    Retorna:
        tuple: (dict, list) - (datos reducidos, varianza explicada)
    """
    
    # Datos por defecto si X es None
    if X is None:
        X = pd.DataFrame({
            "var1": [10, 12, 11, 13, 12, 11, 300, 10],
            "var2": [20, 22, 21, 23, 22, 21, 400, 20],
            "var3": [30, 32, 31, 33, 32, 31, 500, 30]
        })
    
    # Escalar datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    # Retornar 2 elementos
    resultado_dict = {"componentes": X_reducido.tolist()}
    varianza = pca.explained_variance_ratio_.tolist()
    
    return resultado_dict, varianzagit status