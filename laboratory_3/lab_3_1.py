import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris, make_classification

# Загрузка датасета Iris
iris = load_iris()

# Преобразование данных в DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target  # Добавляем столбец с целевыми значениями

# Проверка данных
print("Классы целевой переменной:", iris.target_names)
df.info()

iris.target

# Визуализация данных
# 1. Sepal length vs Sepal width
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
for target, color in zip(range(3), ['red', 'green', 'blue']):
    subset = df[df['target'] == target]
    plt.scatter(subset['sepal length (cm)'], subset['sepal width (cm)'],
                color=color, label=iris.target_names[target], s=50)
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Sepal Length vs Sepal Width')
plt.legend()

# 2. Petal length vs Petal width
plt.subplot(1, 2, 2)
for target, color in zip(range(3), ['red', 'green', 'blue']):
    subset = df[df['target'] == target]
    plt.scatter(subset['petal length (cm)'], subset['petal width (cm)'],
                color=color, label=iris.target_names[target], s=50)
plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Petal Length vs Petal Width')
plt.legend()

plt.tight_layout()
plt.show()

# Визуализация всех признаков с помощью pairplot
sns.pairplot(df, hue='target', vars=iris.feature_names)
plt.show()

# Создание новых датасетов: Setosa и Versicolor (df_1) / Versicolor и Virginica (df_2)
df_1 = df[df['target'].isin([0, 1])]  # Setosa и Versicolor
df_2 = df[df['target'].isin([1, 2])]  # Versicolor и Virginica

# Разделение данных на обучающую и тестовую выборки
X1_train, X1_test, y1_train, y1_test = train_test_split(df_1.drop('target', axis=1), df_1['target'], test_size=0.2, random_state=0)
X2_train, X2_test, y2_train, y2_test = train_test_split(df_2.drop('target', axis=1), df_2['target'], test_size=0.2, random_state=0)

# Обучение моделей логистической регрессии
clf1 = LogisticRegression(random_state=0)
clf1.fit(X1_train, y1_train)
clf2 = LogisticRegression(random_state=0)
clf2.fit(X2_train, y2_train)

# Предсказание и оценка точности для обеих моделей
y1_pred = clf1.predict(X1_test)
y2_pred = clf2.predict(X2_test)

print("Точность модели для Setosa и Versicolor: ", accuracy_score(y1_test, y1_pred))
print("Точность модели для Versicolor и Virginica: ", accuracy_score(y2_test, y2_pred))

# Генерация синтетического набора данных с помощью make_classification
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

# Разделение синтетического датасета на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Обучение модели логистической регрессии на синтетических данных
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Предсказание и вывод точности
y_pred = clf.predict(X_test)
print("Точность модели на синтетических данных: ", accuracy_score(y_test, y_pred))

# Визуализация синтетического набора данных
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='red', label='Класс 0')
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='blue', label='Класс 1')
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.title('Визуализация синтетического набора данных')
plt.legend()
plt.show()