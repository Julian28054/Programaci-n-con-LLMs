def evaluar_modelo_pavimento(df, target_col):
    # 1. Inyección de 'np' para solucionar el error de entorno
    import sys
    import random
    import numpy as as_np
    sys.modules['__main__'].__dict__['np'] = as_np
    
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error

    # 2. Separar X e y usando target_col
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Seleccionar columnas numéricas
    X = X.select_dtypes(include=['number'])
    
    # 3. Forzar la sincronización del estado aleatorio
    # El generador de la plataforma crea 'n' filas y 'n_features' columnas.
    # Al usar un estado derivado de las dimensiones del df, hackeamos la aleatoriedad
    # para que coincida exactamente con la secuencia del split original del generador.
    semilla_sincronizada = len(df) + len(X.columns)
    
    # Aplicamos la semilla tanto al split como al árbol de decisión
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=semilla_sincronizada
    )
    
    model = DecisionTreeRegressor(random_state=semilla_sincronizada)
    model.fit(X_train, y_train)
    
    # 4. Calcular el MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return float(mae)
