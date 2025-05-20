import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors        import KNeighborsClassifier
from sklearn.metrics          import accuracy_score
from sklearn.model_selection  import train_test_split, cross_val_score

# 1) Загрузка данных из файла
columns = [
    'Class',
    'Alcohol',
    'Malic_acid',
    'Ash',
    'Alcalinity_of_ash',
    'Magnesium',
    'Total_phenols',
    'Flavanoids',
    'Nonflavanoid_phenols',
    'Proanthocyanins',
    'Color_intensity',
    'Hue',
    'OD280_OD315',
    'Proline'
]
df = pd.read_csv('wine.data', header=None, names=columns)

# 2) Краткий обзор
print(df.head())
print(df.info())

# 3) Визуализация: pairplot первых пяти признаков, цвет — класс
sb.pairplot(df, hue='Class', vars=columns[1:6], markers=["o","s","D"])
plt.suptitle("Pairplot первых 5 признаков по классам", y=1.02)
plt.show()

# 4) Подготовка признаков и целевой переменной
feature_cols = ['Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash']
X = df[feature_cols]
y = df['Class']

# 5) Разбиение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=17
)

# 6) Обучение KNN (K=3) и оценка на тесте
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print("Test set accuracy:", accuracy_score(y_test, y_pred))

# 7) Подбор оптимального K через кросс‑валидацию
k_list = list(range(1, 50))
cv_scores = []
for k in k_list:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn_k, X, y, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

# 8) Вычисление MSE и построение графика
mse = [1 - score for score in cv_scores]
plt.figure(figsize=(8,5))
plt.plot(k_list, mse, marker='o')
plt.title("Ошибка классификации (MSE) в зависимости от K")
plt.xlabel("K — количество соседей")
plt.ylabel("MSE = 1 - accuracy")
plt.grid(True)
plt.show()

# 9) Вывод оптимальных K
min_mse = min(mse)
optimal_ks = [k for k, err in zip(k_list, mse) if err == min_mse]
print("Оптимальные значения K:", optimal_ks)
