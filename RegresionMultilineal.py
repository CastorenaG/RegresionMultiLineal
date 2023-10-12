import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv("C:\\student_performance.csv")
data = pd.get_dummies(data, columns=['Extracurricular Activities'], drop_first=True)

X = data[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced', 'Extracurricular Activities_Yes']]
y = data['Performance Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Obtener los coeficientes
intercept = model.intercept_  # β0
coefficients = model.coef_  # β1, β2, β3, β4, β5

# Imprimir los coeficientes
print(f'Intercepto (β0): {intercept}')
print(f'Coeficientes (β1, β2, β3, β4, β5): {coefficients}')

# Predecir los valores
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}')
print(f'R²: {r2}')

# Construir la fórmula de regresión lineal
formula = f'Performance Index = {intercept:.2f} + {coefficients[0]:.2f} * Hours Studied + {coefficients[1]:.2f} * Previous Scores + {coefficients[2]:.2f} * Sleep Hours + {coefficients[3]:.2f} * Sample Question Papers Practiced + {coefficients[4]:.2f} * Extracurricular Activities_Yes'
print(f'Fórmula de Regresión Múltiple:')
print(formula)

## Crea la figura para "la prediccion"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(y_test, y_test, color='blue', label='Valores Reales')
ax1.set_xlabel("Valores Reales")
ax1.set_ylabel("Valores Reales Performance Index (Y)")
ax1.set_title("Valores Reales")
ax1.legend()

ax2.scatter(y_test, y_pred, color='green', label='Predicciones')
ax2.set_xlabel("Valores Reales")
ax2.set_ylabel("Predicciones Performance Index ")
ax2.set_title("Predicciones")
ax2.legend()

plt.tight_layout()
plt.show()

## Crea la figura para "Previous Scores"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_test['Previous Scores'], y_test, color='blue', label='Valores Reales')
ax1.set_xlabel("Previous Scores")
ax1.set_ylabel("Valores Reales Performance Index ")
ax1.set_title("Valores Reales")
ax1.legend()

ax2.scatter(X_test['Previous Scores'], y_pred, color='green', label='Predicciones')
ax2.set_xlabel("Previous Scores ")
ax2.set_ylabel("Predicciones Performance Index ")
ax2.set_title("Predicciones")
ax2.legend()

plt.tight_layout()
plt.show()

# Crear la figura para "Sleep hours"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_test['Sleep Hours'], y_test, color='blue', label='Valores Reales')
ax1.set_xlabel("Sleep Hours ")
ax1.set_ylabel("Valores Reales Performance Index ")
ax1.set_title("Valores Reales")
ax1.legend()

ax2.scatter(X_test['Sleep Hours'], y_pred, color='green', label='Predicciones')
ax2.set_xlabel("Sleep Hours ")
ax2.set_ylabel("Predicciones Performance Index")
ax2.set_title("Predicciones")
ax2.legend()

plt.tight_layout()
plt.show()

## Crea la figura para "Sample Question Papers Practiced"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_test['Sample Question Papers Practiced'], y_test, color='blue', label='Valores Reales')
ax1.set_xlabel("Sample Question Papers Practiced")
ax1.set_ylabel("Valores Reales Performance Index ")
ax1.set_title("Valores Reales")
ax1.legend()

ax2.scatter(X_test['Sample Question Papers Practiced'], y_pred, color='green', label='Predicciones')
ax2.set_xlabel("Sample Question Papers Practiced ")
ax2.set_ylabel("Predicciones Performance Index ")
ax2.set_title("Predicciones")
ax2.legend()

plt.tight_layout()
plt.show()

## Crea la figura para "Extracurricular Activities_Yes"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(X_test['Extracurricular Activities_Yes'], y_test, color='blue', label='Valores Reales')
ax1.set_xlabel("Extracurricular Activities_Yes")
ax1.set_ylabel("Valores Reales Performance Index")
ax1.set_title("Valores Reales")
ax1.legend()

ax2.scatter(X_test['Extracurricular Activities_Yes'], y_pred, color='green', label='Predicciones')
ax2.set_xlabel("Extracurricular Activities_Yes")
ax2.set_ylabel("Predicciones Performance Index")
ax2.set_title("Predicciones")
ax2.legend()

plt.tight_layout()
plt.show()
