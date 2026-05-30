import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza=0.95):
    """
    Reduce la dimensionalidad de un dataset usando PCA basado en un umbral de varianza acumulada.
    
    Esta función:
    1. Escala los datos (importante para PCA)
    2. Aplica PCA conservando los componentes necesarios para alcanzar el umbral
    3. Retorna los datos transformados en el nuevo espacio reduzido
    
    Argumentos:
    -----------
    X : pd.DataFrame
        DataFrame con las características originales (filas = muestras, columnas = características)
    umbral_varianza : float
        Valor entre 0 y 1 que indica cuánta varianza queremos conservar (ej. 0.95 = 95%)
    
    Retorna:
    --------
    numpy.ndarray
        Array con las nuevas coordenadas de los datos en el espacio reducido
    """
    
    # Paso 1: Escalar los datos
    # Importante: PCA requiere datos estandarizados para funcionar correctamente
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Paso 2: Aplicar PCA con el umbral de varianza
    # El parámetro n_components como float representa la proporción de varianza a explicar
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    # Paso 3: Retornar los datos reducidos
    return X_reducido
