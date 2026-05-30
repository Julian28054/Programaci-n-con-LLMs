def comprimir_dimensiones_por_varianza(X, umbral_varianza):
    """
    Retorna: numpy.ndarray con las coordenadas reducidas
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    return X_reducido  # Retorna 1 array