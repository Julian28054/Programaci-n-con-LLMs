import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def comprimir_dimensiones_por_varianza(*args, **kwargs):

    X = kwargs.get("X", kwargs.get("componentes"))
    umbral_varianza = kwargs.get("umbral_varianza")

    X = pd.DataFrame(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=umbral_varianza)

    return pca.fit_transform(X_scaled)
