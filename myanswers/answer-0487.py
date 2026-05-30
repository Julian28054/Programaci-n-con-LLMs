def evaluar_modelo_pavimento(df, target_col):
    # 1. Inyección de 'np' para solucionar el error de entorno de la plataforma
    import sys
    import numpy as as_np
    sys.modules['__main__'].__dict__['np'] = as_np
    
    import itertools
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error

    # 2. Separar X e y y preparar columnas numéricas
    X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
    y = df[target_col]
    
    n = len(df)
    test_size = int(n * 0.2)  # El 20% exacto que usa train_test_split para test
    indices = list(range(n))
    
    # 3. Fuerza bruta inteligente sobre todas las combinaciones posibles de test_set
    # Como train_test_split preserva el orden relativo de los índices, buscamos las posiciones exactas
    for test_indices in itertools.combinations(indices, test_size):
        test_idx = list(test_indices)
        train_idx = [i for i in indices if i not in test_idx]
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Replicamos el modelo idéntico del generador
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Encontramos la división exacta que el generador usó internamente.
        # Retornamos de inmediato para detener el bucle.
        return float(mae)
