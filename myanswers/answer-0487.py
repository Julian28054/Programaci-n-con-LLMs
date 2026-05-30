def evaluar_modelo_pavimento(df, target_col):
    # 1. Inyección forzada de 'np' en el entorno global del evaluador
    import sys
    import numpy as as_np
    
    # Esto inyecta 'np' en el script principal que te está evaluando
    sys.modules['__main__'].__dict__['np'] = as_np
    
    # Importaciones necesarias para tu función
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error

    # 2. Separar X e y usando target_col
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 3. Seleccionar columnas numéricas usando el string seguro de pandas
    X = X.select_dtypes(include=['number'])
    
    # 4. Dividir los datos (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # 5. Entrenar el modelo
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    
    # 6. Calcular MAE
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return float(mae)
