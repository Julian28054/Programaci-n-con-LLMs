def evaluar_modelo_pavimento(df, target_col):
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    import numpy as np
    
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.select_dtypes(include=[np.number])  # Asumiendo que numpy está disponible
    
    # Fijar una semilla para que la división sea determinística
    np.random.seed(42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
