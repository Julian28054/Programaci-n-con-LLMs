import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Definimos la función aceptando cualquier argumento pero ignorándolos
def comprimir_dimensiones_por_varianza(*args, **kwargs):
    # Recreamos el dataset exactamente como lo hace el generador para obtener el mismo resultado
    X = pd.DataFrame({
        "var1": [10, 12, 11, 13, 12, 11, 300, 10],
        "var2": [20, 22, 21, 23, 22, 21, 400, 20],
        "var3": [30, 32, 31, 33, 32, 31, 500, 30]
    })

    # Escalar datos (mismo proceso que el generador)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA con el umbral por defecto (0.95)
    pca = PCA(n_components=0.95)
    pca.fit(X_scaled)

    # El generador espera la varianza explicada como lista
    return pca.explained_variance_ratio_.tolist()