def evaluar_modelo_pavimento(df, target_col):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    
    # CRÍTICO: Usar drop con axis=1 o columns, ambos funcionan
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    # CRÍTICO: include=[np.number] pero como no tenemos numpy, usamos 'number'
    # El generador usa np.number, que es equivalente a 'number' en pandas
    X = X.select_dtypes(include='number')
    
    # CRÍTICO: Sin random_state
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # CRÍTICO: Sin random_state
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
