import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_recall_curve, RocCurveDisplay, roc_curve
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz

df = pd.read_csv('diabetes.csv')
df.info()

# Разделение данных на признаки и целевую переменную
X = df.drop(columns=['Outcome'])  # Все столбцы, кроме 'Outcome', это признаки
y = df['Outcome']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Модель логистической регрессии
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
y_pred_logistic = logistic_model.predict(X_test)

# Модель решающего дерева
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

# Метрики для логистической регрессии
print("Логистическая регрессия:")
print(classification_report(y_test, y_pred_logistic))
print("Accuracy:", accuracy_score(y_test, y_pred_logistic))

# Метрики для решающего дерева
print("\nРешающее дерево:")
print(classification_report(y_test, y_pred_tree))
print("Accuracy:", accuracy_score(y_test, y_pred_tree))

# Исследуем разные значения глубины дерева для оптимизации
depths = np.arange(1, 21)
accuracies = []

for depth in depths:
    tree_model = DecisionTreeClassifier(max_depth=depth, random_state=42)  # Контроль рандомизации для воспроизводимости
    tree_model.fit(X_train, y_train)
    y_pred = tree_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# Построение графика зависимости точности от глубины дерева
plt.figure(figsize=(8, 6))
plt.plot(depths, accuracies, marker='o')
plt.xlabel("Depth of Decision Tree")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. Depth of Decision Tree")
plt.grid(True)
plt.show()


# Визуализация дерева решений с помощью Graphviz
dot_data = export_graphviz(tree_model, out_file=None,
                           feature_names=X.columns,
                           class_names=['Non-Diabetic', 'Diabetic'],
                           filled=True, rounded=True,
                           special_characters=True)

# Отображение дерева решений
graph = graphviz.Source(dot_data)
graph.render("diabetes_tree")  # Сохранение в файл

# Важность признаков для модели решающего дерева
feature_importances = pd.Series(tree_model.feature_importances_, index=X.columns)

# Визуализация важности признаков
plt.figure(figsize=(8, 6))
feature_importances.nlargest(10).plot(kind='barh', color='skyblue')
plt.xlabel('Важность признака')
plt.ylabel('Признак')
plt.title('Важность признаков для решающего дерева')
plt.show()

# Precision-Recall кривая
y_scores = tree_model.predict_proba(X_test)[:, 1]  # Вероятности для положительного класса (1)
precision, recall, _ = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='b', label='PR кривая')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall кривая')
plt.legend()
plt.show()

# ROC кривая и AUC
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr)
roc_display.plot()
plt.title('ROC кривая')
plt.show()