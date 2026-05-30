def evaluar_modelo_pavimento(df, target_col):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    
    # 1. Separar X e y
    y = df[target_col]
    X = df.drop(target_col, axis=1)
    
    # 2. Seleccionar solo columnas numéricas (manualmente)
    columnas_numericas = []
    for col in X.columns:
        if X[col].dtype in ['float64', 'float32', 'int64', 'int32']:
            columnas_numericas.append(col)
    X = X[columnas_numericas]
    
    # 3. Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 4. Entrenar
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 5. Evaluar
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
