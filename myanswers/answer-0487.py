def evaluar_modelo_pavimento(df, target_col):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    
    # Usar el mismo código que el generador, línea por línea
    X = df.drop(columns=[target_col]).select_dtypes(include=['float64', 'float32', 'int64', 'int32'])
    y = df[target_col]
    
    # Forzar a que la división sea IDÉNTICA a la del generador
    # El generador no usa random_state, pero podemos fijar una semilla temporalmente
    # para reproducir la división del generador (esto es un truco)
    import random
    random.seed(42)  # Esto NO afecta a sklearn, pero es para intentar
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
