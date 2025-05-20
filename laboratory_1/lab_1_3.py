import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

# Загрузка набора данных "diabetes"
diabetes = datasets.load_diabetes(as_frame=True)

# Преобразуем в DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)

# Целевая переменная (сахарный диабет)
target = diabetes.target

# Исключаем признак "sex" из анализа
features = diabetes['feature_names']
features.remove('sex')

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(df, target, test_size=0.2, random_state=0)

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание
predictions = model.predict(X_test)

# Получаем метрики
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mape = mean_absolute_percentage_error(y_test, predictions)

# Вывод метрик
print("\nОценка качества модели:")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Coefficient of Determination (R2): {r2:.2f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")

# Итоги предсказания
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print("\nСравнение предсказанных и реальных значений:")
print(df_pred.head())

# Визуализация предсказанных и истинных значений
plt.figure(figsize=(10, 6))
plt.plot(y_test, predictions, 'o')
plt.plot([-100, 100], [-100, 100], color='black', linestyle='--')  # Линия y = x для ориентира
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Сравнение истинных значений и предсказаний')
plt.grid(True)
plt.show()