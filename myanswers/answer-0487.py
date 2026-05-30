import pandas as pd
import numpy as as_np  # Lo importamos con alias temporal para evitar conflictos
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Forzamos la existencia de 'np' en el espacio global del script
globals()['np'] = as_np

def evaluar_modelo_pavimento(df, target_col):
    # Forzamos la existencia de 'np' también dentro del espacio local de la función
    import numpy as np
    globals()['np'] = np
    
    # 1. Separar X e y usando target_col
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 2. Seleccionar solo columnas numéricas (usamos el string compatible por si acaso)
    X = X.select_dtypes(include=['number'])
    
    # 3. Dividir los datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 4. Entrenar un modelo DecisionTreeRegressor de sklearn
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 5. Calcular el error absoluto medio (MAE)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return float(mae)
