import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# Загрузка набора данных "diabetes"
diabetes = datasets.load_diabetes(as_frame=True)

# Преобразуем в DataFrame
df = pd.DataFrame(data=diabetes.data, columns=diabetes.feature_names)
# Информация о данных
print("\nИнформация о данных:")
df.info()

# Целевая переменная (сахарный диабет)
target = diabetes.target
target

# Исключаем признак "sex" из анализа
features = diabetes['feature_names']
features.remove('sex')

# Построение графиков для признаков
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Diabetes Dataset - Фичи и Цель')

for i in range(3):
    for j in range(3):
        n = j + i * 3
        feature = features[n]
        axs[i, j].scatter(diabetes['data'][feature], diabetes['target'], s=10)
        axs[i, j].set_xlabel(feature)
        axs[i, j].set_ylabel('Target')
        axs[i, j].grid(True)

plt.tight_layout()
plt.show()

# Модель данных
X = df  # Признаки
Y = target  # Цель

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# Создание модели линейной регрессии
model = LinearRegression()

# Обучение модели
model.fit(X_train, y_train)

# Предсказание
predictions = model.predict(X_test)

# Итоги предсказания
df_pred = pd.DataFrame({'Actual': y_test, 'Predicted': predictions})
print("\nСравнение предсказанных и реальных значений:")
print(df_pred)

# Получаем коэффициенты модели
coef = model.coef_
intercept = model.intercept_

print("\nКоэффициенты модели:")
print("Наклон:", coef)
print("Пересечение:", intercept)

print("\nРазмеры данных (обучение):")
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# Построение графика предсказаний против реальных значений
plt.figure(figsize=(10, 6))

# Строим регрессионную прямую
plt.plot(y_test, predictions, 'o')

# Добавляем метки и линию y=x для сравнения
plt.plot([-100, 100], [-100, 100], color='black', linestyle='--')  # Линия y = x для ориентира
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.title('Сравнение истинных значений и предсказаний')
plt.grid(True)

# Отображаем график
plt.show()

# Построение регрессионной линии для каждого признака
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Линейная регрессия для каждого признака')

index = 0
for i in range(3):
    for j in range(3):
        # Пропускаем признак "sex"
        if index == 1:
            index += 1

        feature_name = X.columns[index]
        X_train_selected = X_train.iloc[:, [index]]
        X_test_selected = X_test.iloc[:, [index]]

        # Обучение модели для одного признака
        model.fit(X_train_selected, y_train)

        # Построение регрессионной линии
        axs[i, j].scatter(X_test_selected, y_test, color='blue', label='Данные')  # Данные
        axs[i, j].plot(X_test_selected, model.predict(X_test_selected), color='red', label='Регрессия')  # Линия регрессии
        axs[i, j].set_xlabel(feature_name)
        axs[i, j].set_ylabel('Target')
        axs[i, j].legend()

        index += 1

plt.tight_layout()
plt.show()

# Функция наименьших квадратов для одного признака
def least_squares_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Вычисляем коэффициенты регрессии
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    b1 = numerator / denominator  # Наклон
    b0 = y_mean - b1 * x_mean  # Свободный член

    return b0, b1

# Построение графиков регрессии для каждого признака с использованием least squares
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
fig.suptitle('Регрессия на основе наименьших квадратов')

index = 0
for i in range(3):
    for j in range(3):
        # Пропускаем признак "sex"
        if index == 1:
            index += 1

        feature_name = X.columns[index]

        # Выбор данных по одному признаку
        X_train_selected = X_train.iloc[:, [index]].squeeze()
        y_train_selected = y_train

        # Вычисление коэффициентов
        b0, b1 = least_squares_regression(X_train_selected, y_train_selected)

        # Построение графика
        axs[i, j].scatter(X_train_selected, y_train_selected, color='blue', label='Данные')
        axs[i, j].plot(X_train_selected, b0 + b1 * X_train_selected, color='red', label='Регрессия')
        axs[i, j].set_xlabel(feature_name)
        axs[i, j].set_ylabel('Target')
        axs[i, j].grid(True)
        axs[i, j].legend()

        index += 1

plt.tight_layout()
plt.show()