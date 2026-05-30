import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(X, umbral_varianza):
    """
    Reduce la dimensionalidad de un dataset usando PCA basado en un umbral de varianza acumulada.
    
    Parameters:
    -----------
    X : pd.DataFrame
        DataFrame con las características originales (genes/variables)
    umbral_varianza : float
        Valor entre 0 y 1 indicando la varianza acumulada a conservar (ej. 0.95 para 95%)
    
    Returns:
    --------
    numpy.ndarray
        Array con las coordenadas de los datos en el espacio reducido
    """
    
    # Validar el umbral
    if not 0 <= umbral_varianza <= 1:
        raise ValueError("El umbral de varianza debe estar entre 0 y 1")
    
    # Estandarizar los datos (centrar y escalar)
    # Esto es crucial para PCA porque las variables con escalas diferentes
    # podrían dominar el análisis incorrectamente
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Aplicar PCA sin especificar componentes, calculamos todos
    pca = PCA()
    pca.fit(X_scaled)
    
    # Calcular varianza acumulada
    varianza_acumulada = np.cumsum(pca.explained_variance_ratio_)
    
    # Encontrar cuántos componentes necesitamos para alcanzar el umbral
    n_componentes = np.argmax(varianza_acumulada >= umbral_varianza) + 1
    
    # Re-entrenar PCA con el número óptimo de componentes
    pca_optimo = PCA(n_components=n_componentes)
    X_reducido = pca_optimo.fit_transform(X_scaled)
    
    return X_reducido
