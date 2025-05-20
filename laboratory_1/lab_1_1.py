import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных из CSV
df = pd.read_csv('student_scores.csv')

# Информация о наборе данных
print("\nИнформация о наборе данных:")
df.info()

# Первые несколько строк данных
print("\nПервые 5 строк данных:")
print(df.head())

print("\nМинимальные значения по столбцам:")
print(df.min())

print("\nМаксимальные значения по столбцам:")
print(df.max())

print("\nМедианные значения по столбцам:")
print(df.median())

# Построение исходных данных
plt.figure(figsize=(8, 6))
plt.xlabel('Scores')
plt.ylabel('Hours')
plt.title('Зависимость Hours от Scores')
plt.grid(True)
plt.scatter(df['Scores'], df['Hours'], c='black', label='Исходные данные')
plt.legend()
plt.show()

# Функция для вычисления коэффициентов линейной регрессии методом наименьших квадратов
def least_squares_regression(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Вычисление коэффициентов линейной регрессии
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    b1 = numerator / denominator  # коэффициент наклона
    b0 = y_mean - b1 * x_mean     # свободный член

    return b0, b1

# Данные
x = df["Hours"]
y = df["Scores"]

# Пользовательский ввод для выбора переменных
user_input = input("Выберите переменную X - Hours[h] or Scores[s]: ").strip().lower()
while user_input not in ['h', 's']:
    user_input = input("Неправильный ввод, попробуйте снова (h/s): ").strip().lower()

# Меняем переменные, если выбраны "Scores"
if user_input == 's':
    x = df["Scores"]
    y = df["Hours"]

# Вычисление параметров регрессии
b0, b1 = least_squares_regression(x, y)

# Вывод коэффициентов регрессии
print("\nПараметры регрессионной прямой:")
print(f"Свободный член (b0) = {b0}")
print(f"Коэффициент наклона (b1) = {b1}")

# Построение графика исходных данных и регрессионной линии
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='black', label='Исходные данные')
plt.plot(x, b0 + b1 * x, color='blue', label='Регрессионная прямая')
plt.xlabel(x.name)
plt.ylabel(y.name)
plt.title('Линейная регрессия')
plt.legend()
plt.grid(True)
plt.show()

# Вычисление предсказанных значений
predicted_y = b0 + b1 * x

# Визуализация исходных данных и регрессионной прямой с квадратами ошибок
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='black', label='Исходные данные')
plt.plot(x, predicted_y, color='blue', label='Регрессионная прямая')

# Визуализация квадратов ошибок
for i in range(len(x)):
    # Определение положения прямоугольников для визуализации ошибок
    if predicted_y[i] > y[i]:  # Точки над линией
        rect = plt.Rectangle((x[i], y[i]), predicted_y[i] - y[i], predicted_y[i] - y[i], color='green', alpha=0.3)
    else:  # Точки под линией
        rect = plt.Rectangle((x[i], predicted_y[i]), y[i] - predicted_y[i], y[i] - predicted_y[i], color='green', alpha=0.3)
    plt.gca().add_patch(rect)

plt.xlabel(x.name)
plt.ylabel(y.name)
plt.title('Линейная регрессия с визуализацией ошибок')
plt.legend()
plt.grid(True)
plt.show()