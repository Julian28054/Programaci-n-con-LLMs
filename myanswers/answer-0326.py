import pandas as pd
from sklearn.decomposition import PCA

def comprimir_dimensiones_por_varianza(componentes):
    pca = PCA()
    pca.fit(componentes)
    return pca.explained_variance_ratio_.tolist()
