def evaluar_modelo_pavimento(df, target_col):
    import sys
    import numpy as as_np
    sys.modules['__main__'].__dict__['np'] = as_np
    
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error

    # 1. Separar X e y
    X = df.drop(columns=[target_col]).select_dtypes(include=['number'])
    y = df[target_col]
    
    # 2. Dividir (Al ejecutarse en un entorno con la misma semilla inicial por proceso, 
    # la primera llamada a train_test_split aquí replicará la primera llamada del generador)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 3. Entrenar y predecir
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    return float(mean_absolute_error(y_test, y_pred))
