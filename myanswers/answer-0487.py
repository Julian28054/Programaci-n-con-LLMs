import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def evaluar_modelo_pavimento(df, target_col):
    """
    Evalúa un modelo de regresión para pavimentos.
    
    Lógica:
    1. Filtra solo columnas numéricas para evitar errores en el modelo.
    2. Separa el target del resto de los features.
    3. Realiza una partición 80/20.
    4. Entrena un DecisionTreeRegressor estándar.
    5. Retorna el MAE como valor flotante.
    """
    
    # 1. Separación de datos y selección de columnas numéricas
    # Nos aseguramos de trabajar solo con datos que el árbol puede procesar
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]
    
    # 2. División Entrenamiento/Prueba (80/20)
    # Usamos un split estándar. Si esto fuera un entorno de producción,
    # incluiríamos un random_state para reproducibilidad.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 3. Entrenamiento del modelo
    # El DecisionTreeRegressor es ideal para capturar relaciones no lineales 
    # en datos de ingeniería civil.
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Predicción y Cálculo del MAE
    # El MAE es la métrica más interpretable para un ingeniero:
    # nos dice cuánto se desvía en promedio la predicción respecto a la realidad.
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    
    return float(mae)