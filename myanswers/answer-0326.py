import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

def comprimir_dimensiones_por_varianza(X=None, umbral_varianza=0.95, componentes=None):

    datos = componentes if componentes is not None else X

    pca = PCA()
    pca.fit(datos)

    return pca.explained_variance_ratio_.tolist()
