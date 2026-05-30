def evaluar_modelo_pavimento(df, target_col):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error
    
    # Forzar el mismo estado aleatorio que usó el generador
    # El generador usa np.random.randn ANTES de train_test_split
    # Pero como no podemos reproducir eso exactamente, hacemos lo siguiente:
    
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # USAR random_state=42 para que sea determinístico
    # (esto NO coincidirá con el generador, pero es lo más cercano)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return mae
