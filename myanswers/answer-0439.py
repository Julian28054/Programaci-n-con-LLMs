import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def detectar_por_reconstruccion(X_train, X_test, umbral_percentil):
    # 1. Escalar los datos
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2. PCA conservando el 90% de la varianza
    pca = PCA(n_components=0.90)
    pca.fit(X_train_scaled)

    # Reconstrucción de train
    X_train_proj = pca.transform(X_train_scaled)
    X_train_rec = pca.inverse_transform(X_train_proj)

    # Reconstrucción de test
    X_test_proj = pca.transform(X_test_scaled)
    X_test_rec = pca.inverse_transform(X_test_proj)

    # 3. Error cuadrático medio (MSE) por fila
    errores_train = np.mean((X_train_scaled - X_train_rec) ** 2, axis=1)
    errores_test = np.mean((X_test_scaled - X_test_rec) ** 2, axis=1)

    # 4. Umbral basado en el percentil de los errores de entrenamiento
    umbral = np.percentile(errores_train, umbral_percentil)

    # Detectar anomalías
    return errores_test > umbral
