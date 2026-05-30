def evaluar_modelo_pavimento(df, target_col):
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import mean_absolute_error

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    # Seed reproducible basado en el contenido exacto del df
    seed = hash(df.to_json()) % (2**31)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )

    model = DecisionTreeRegressor(random_state=seed)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return mean_absolute_error(y_test, y_pred)
