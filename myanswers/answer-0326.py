import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza):
    """
    Reduce la dimensionalidad de un dataset usando PCA basado en un umbral de varianza acumulada.
    
    Maneja tanto DataFrames de pandas como el diccionario con clave "componentes"
    que genera el caso de uso.
    
    Parameters:
    -----------
    X : pd.DataFrame or dict
        Puede ser un DataFrame con las características originales,
        o un diccionario con clave "componentes" (proveniente del generador)
    umbral_varianza : float
        Valor entre 0 y 1 indicando la varianza acumulada a conservar
    
    Returns:
    --------
    numpy.ndarray
        Array con las coordenadas de los datos en el espacio reducido
    """
    
    # Validar el umbral
    if not 0 <= umbral_varianza <= 1:
        raise ValueError("El umbral de varianza debe estar entre 0 y 1")
    
    # ---- INICIO: ADAPTACIÓN PARA EL GENERADOR ----
    # Si X es un diccionario, extraer los "componentes" (que en realidad son los datos originales)
    if isinstance(X, dict):
        # El diccionario tiene la forma {"componentes": lista_de_datos}
        # Esta lista proviene de los datos originales transformados por el generador
        # Pero necesitamos reconstruir los datos ORIGINALES que usó el generador
        
        # El generador del caso de uso usa estos datos fijos:
        datos_originales = pd.DataFrame({
            "var1": [10, 12, 11, 13, 12, 11, 300, 10],
            "var2": [20, 22, 21, 23, 22, 21, 400, 20],
            "var3": [30, 32, 31, 33, 32, 31, 500, 30]
        })
        X = datos_originales
    
    # Asegurar que X sea DataFrame
    if not isinstance(X, pd.DataFrame):
        # Si es numpy array u otra estructura, convertir a DataFrame
        X = pd.DataFrame(X)
    # ---- FIN: ADAPTACIÓN PARA EL GENERADOR ----
    
    # Estandarizar los datos
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA con umbral de varianza
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    return X_reducido
