'''
Momento de Retroalimentación: 
Módulo 2 Uso de framework o biblioteca de aprendizaje máquina para la implementación de una solución

Viviana Alanis Fraige | A01236316
9/7/2024
'''

# Librerias a utilizar
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt

'''IMPORTAR DATOS A UTILIZAR: se utilzara el dataset de cancer para predecir 
el nivel de cancer de un paciente'''
# Cargar el conjunto de datos
df = pd.read_csv('Cancer_Data.csv')
print(df.head())


'''PREPROCESAMIENTO DE DATOS'''
# Eliminar la columna 'id' ya que no es útil para el análisis
df = df.drop(columns=['id'])

# Convertir la columna 'diagnosis' a valores binarios: 'M' -> 1, 'B' -> 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})

# Tratar valores NaN: Imputar con la media de la columna
df = df.fillna(df.mean())

# Seleccionar las características y la variable objetivo
X = df[['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean']]
y = df['diagnosis']


'''ENTRENAMIENTO Y PRUEBA'''
# Separar los datos en Train/Validation/Test
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Verificar las dimensiones de los conjuntos de datos
print(f"Tamaño de X_train: {X_train.shape}")
print(f"Tamaño de X_val: {X_val.shape}")
print(f"Tamaño de X_test: {X_test.shape}")


'''IMPLEMENTANDO REGRESION LINEAL'''
# Entrenar el modelo de Regresión Lineal Simple
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones en los conjuntos de validación y prueba
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

# Evaluación del desempeño
mse_val = mean_squared_error(y_val, y_val_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
y_val_pred_binary = (y_val_pred > 0.5).astype(int)
y_test_pred_binary = (y_test_pred > 0.5).astype(int)
accuracy_val = accuracy_score(y_val, y_val_pred_binary)
accuracy_test = accuracy_score(y_test, y_test_pred_binary)

print(f'MSE en Validación: {mse_val}')
print(f'MSE en Prueba: {mse_test}')
print(f'Accuracy en Validación: {accuracy_val}')
print(f'Accuracy en Prueba: {accuracy_test}')

# Diagnóstico visual: Gráfico de Predicciones vs Valores Reales
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(y_val, y_val_pred, color='blue', label='Validación')
plt.scatter(y_test, y_test_pred, color='green', label='Prueba')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.legend()
plt.title('Predicciones vs Valores Reales (Regresión Lineal Simple)')


'''IMPLEMENTANDO RIDGE'''
# Implementar Ridge Regularization para mejorar el desempeño
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

# Predicciones con Ridge
y_val_pred_ridge = ridge_model.predict(X_val)
y_test_pred_ridge = ridge_model.predict(X_test)

# Evaluación con Ridge
mse_val_ridge = mean_squared_error(y_val, y_val_pred_ridge)
mse_test_ridge = mean_squared_error(y_test, y_test_pred_ridge)
y_val_pred_ridge_binary = (y_val_pred_ridge > 0.5).astype(int)
y_test_pred_ridge_binary = (y_test_pred_ridge > 0.5).astype(int)
accuracy_val_ridge = accuracy_score(y_val, y_val_pred_ridge_binary)
accuracy_test_ridge = accuracy_score(y_test, y_test_pred_ridge_binary)

print(f'MSE en Validación con Ridge: {mse_val_ridge}')
print(f'MSE en Prueba con Ridge: {mse_test_ridge}')
print(f'Accuracy en Validación con Ridge: {accuracy_val_ridge}')
print(f'Accuracy en Prueba con Ridge: {accuracy_test_ridge}')

# Comparación gráfica: Ridge vs. Lineal Simple
plt.subplot(1, 2, 2)
plt.scatter(y_val, y_val_pred_ridge, color='red', label='Ridge Validación')
plt.scatter(y_test, y_test_pred_ridge, color='orange', label='Ridge Prueba')
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.legend()
plt.title('Predicciones vs Valores Reales (Ridge)')
plt.show()

# Diagnóstico de sesgo, varianza y ajuste
def diagnose_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    print(f"MSE: {mse}")
    if mse < 0.1:
        bias = "Bajo"
    elif mse < 0.2:
        bias = "Medio"
    else:
        bias = "Alto"
    return bias

print("Diagnóstico del Modelo Lineal Simple:")
bias_simple = diagnose_model(y_val, y_val_pred)
print(f"Sesgo: {bias_simple}")

print("Diagnóstico del Modelo Ridge:")
bias_ridge = diagnose_model(y_val, y_val_pred_ridge)
print(f"Sesgo Ridge: {bias_ridge}")

# Evaluación adicional para varianza
def variance_diagnosis(y_train, y_train_pred, y_val, y_val_pred):
    train_mse = mean_squared_error(y_train, y_train_pred)
    val_mse = mean_squared_error(y_val, y_val_pred)
    print(f"MSE en Entrenamiento: {train_mse}")
    print(f"MSE en Validación: {val_mse}")
    if train_mse < val_mse:
        variance = "Alto"
    else:
        variance = "Bajo"
    return variance

print("Diagnóstico de Varianza del Modelo Lineal Simple:")
variance_simple = variance_diagnosis(y_train, model.predict(X_train), y_val, y_val_pred)
print(f"Varianza: {variance_simple}")

print("Diagnóstico de Varianza del Modelo Ridge:")
variance_ridge = variance_diagnosis(y_train, ridge_model.predict(X_train), y_val, y_val_pred_ridge)
print(f"Varianza Ridge: {variance_ridge}")