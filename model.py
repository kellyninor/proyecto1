# Importación librerías
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from joblib import dump

# Carga de datos de archivo .csv
dataTraining = pd.read_csv('https://raw.githubusercontent.com/davidzarruk/MIAD_ML_NLP_2023/main/datasets/dataTrain_carListings.zip')

# Crear función para calcular variables
def calcularEdadVehiculo (data):
    # Calcular la edad del vehículo
    current_year = datetime.datetime.now().year
    data['Age'] = current_year - data['Year']
    return data

data = calcularEdadVehiculo(dataTraining)

features = data.drop(columns=['Price'])  # Features

# Escalar features
numeric_features = ['Mileage', 'Age','Year']
categorical_features = [ 'State', 'Make', 'Model']

# Definir preprocesamiento para variables numéricas y categóricas
numeric_transformer = Pipeline(steps=[
    ('scaler', MinMaxScaler())  # Cambiar a MinMaxScaler
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
        remainder='passthrough'  # Mantener el resto de las columnas sin cambiosç
    )

# Aplicar el preprocesamiento al conjunto de datos
processed_data = preprocessor.fit_transform(features)

# Convertir la matriz dispersa a una matriz densa (opcional)
processed_data_dense = processed_data.toarray()


# Extraer la columna 'Price' como un array 1D
price_values = data['Price'].values.reshape(-1, 1)

# Inicializar y aplicar el MinMaxScaler solo a la columna 'Price'
scaler = MinMaxScaler()
price_scaled = scaler.fit_transform(price_values)

# Dividir los datos en características (X) y variable objetivo (y)
X = processed_data_dense
y = price_scaled

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Imprimir las dimensiones de los conjuntos de entrenamiento y prueba
print("Dimensiones del conjunto de entrenamiento X:", X_train.shape)
print("Dimensiones del conjunto de prueba X:", X_test.shape)
print("Dimensiones del conjunto de entrenamiento y:", y_train.shape)
print("Dimensiones del conjunto de prueba y:", y_test.shape)

# Inicializar el modelo de árbol de decisión
decision_tree = DecisionTreeRegressor(random_state=42)

# Entrenar el modelo
decision_tree.fit(X_train, y_train)

# Predicción en el conjunto de entrenamiento y prueba
y_train_pred = decision_tree.predict(X_train)
y_test_pred = decision_tree.predict(X_test)

# Evaluar el rendimiento del modelo
ds_train_mse = mean_squared_error(y_train, y_train_pred)
ds_test_mse = mean_squared_error(y_test, y_test_pred)
ds_train_r2 = r2_score(y_train, y_train_pred)
ds_test_r2 = r2_score(y_test, y_test_pred)

print("Error cuadrático medio (MSE) en conjunto de entrenamiento:", ds_train_mse)
print("Error cuadrático medio (MSE) en conjunto de prueba:", ds_test_mse)
print("Coeficiente de determinación (R^2) en conjunto de entrenamiento:", ds_train_r2)
print("Coeficiente de determinación (R^2) en conjunto de prueba:", ds_test_r2)


# Nombre de archivo para guardar el modelo
file_name = 'decision_tree.pkl'

# Guardar el modelo como un archivo .pkl
dump(decision_tree, file_name)

print("Modelo guardado como:", file_name)

dump(preprocessor, 'preprocessor.pkl')

dump(scaler, 'scaler.pkl')