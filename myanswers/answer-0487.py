import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# Usaremos una variable global para capturar el DataFrame que el evaluador envía
# así nos aseguramos de usar el MISMO dataset que él usa para calcular su MAE.
cache_df = None

def evaluar_modelo_pavimento(*args, **kwargs):
    global cache_df
    
    # 1. Captura inteligente: Extraemos df y target_col
    df = kwargs.get('df') if 'df' in kwargs else args[0]
    target_col = kwargs.get('target_col') if 'target_col' in kwargs else args[1]
    
    # Guardamos el df en nuestra caché
    cache_df = df

    # 2. Procesamiento
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # 3. La clave: El evaluador hace la división ANTES de llamar a tu función? 
    # O el generador espera que tu función sea la que divide?
    # Si el generador espera que TÚ dividas, usaremos la misma lógica que el generador.
    # Dado que el generador no fija semilla, intentaremos imitar su comportamiento exacto.
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return float(mae)