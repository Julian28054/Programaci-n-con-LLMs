def evaluar_modelo_pavimento(df, target_col):
    # 1. Separar X e y usando target_col
    # 2. Seleccionar solo columnas numéricas
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 3. Dividir los datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 4. Entrenar un modelo DecisionTreeRegressor
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 5. Calcular el error absoluto medio (MAE) en el conjunto de prueba
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    # Salida: valor del MAE como float
    return float(mae)
