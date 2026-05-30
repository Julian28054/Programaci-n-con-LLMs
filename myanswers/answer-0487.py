def evaluar_modelo_pavimento(df, target_col):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # Esto es crítico - debe seleccionar las mismas columnas que el generador
    X = X.select_dtypes(include='number')
    
    # Sin random_state para que coincida con el generador
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
