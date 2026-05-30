def evaluar_modelo_pavimento(df, target_col, random_state=None):
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # Pasamos el random_state a la división
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Pasamos el random_state al modelo
    model = DecisionTreeRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    return float(mean_absolute_error(y_test, y_pred))
