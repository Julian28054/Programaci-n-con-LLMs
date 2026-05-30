import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza=0.95, componentes=None):
    """
    Reduce la dimensionalidad usando PCA basado en un umbral de varianza acumulada.
    
    Argumentos:
    -----------
    X : pd.DataFrame
        DataFrame con las características originales
    umbral_varianza : float
        Valor entre 0 y 1 (ej. 0.95 para 95%)
    componentes : any
        Parámetro de compatibilidad (ignorado)
    
    Retorna:
    --------
    tuple: (dict, list)
        - diccionario con las coordenadas reducidas
        - lista con la varianza explicada por cada componente
    """
    
    # 1. Escalar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Aplicar PCA
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    # 3. Retornar 2 elementos: diccionario + varianza explicada
    resultado_dict = {
        "componentes": X_reducido.tolist()
    }
    
    varianza_explicada = pca.explained_variance_ratio_.tolist()
    
    return resultado_dict, varianza_explicada
