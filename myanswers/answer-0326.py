def comprimir_dimensiones_por_varianza(componentes, umbral_varianza: float):
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(componentes)
    
    pca = PCA(n_components=umbral_varianza)
    X_reducido = pca.fit_transform(X_scaled)
    
    return X_reducido, pca.explained_variance_ratio_.tolist()