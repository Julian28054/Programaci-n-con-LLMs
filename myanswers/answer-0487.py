def evaluar_modelo_pavimento(df, target_col):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    
    print("Columnas originales:", df.columns.tolist())
    print("Tipos de datos:", df.dtypes)
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X = X.select_dtypes(include='number')
    print("Columnas numéricas seleccionadas:", X.columns.tolist())
    print("Shape de X:", X.shape)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    print("Shapes - train:", X_train.shape, "test:", X_test.shape)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    print("MAE calculado:", mae)
    
    return mae
