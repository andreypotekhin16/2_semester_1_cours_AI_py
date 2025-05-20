import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Загрузка данных
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
    'OD280/OD315_of_diluted_wines',
    'Proline'
]
data = pd.read_csv("wine.data", header=None, names=columns)

#--------------------------------ниже код---------------------------------------------

numeric_cols = [c for c in data.columns if c != columns[0]]
n = len(numeric_cols)
ncols = 4
nrows = math.ceil(n / ncols)

# 2. Собираем все гистограммы в одну фигуру
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.histplot(data[col], bins=30, ax=ax, color='mediumpurple', edgecolor='black')
    ax.set_title(col)
    ax.grid(True)
# удаляем лишние подплоты
for ax in axes.flatten()[n:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.show()

# 3. Собираем все boxplot’ы количественных признаков в одну фигуру
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(y=data[col], ax=ax, color='lightblue')
    ax.set_title(col)
# удаляем лишние подплоты
for ax in axes.flatten()[n:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.show()

# 4. Countplot по классам (отдельно — он всего один)
plt.figure(figsize=(6, 4))
sns.countplot(x=columns[0], data=data, palette='Set2')
plt.title("Распределение по классам ")
plt.xlabel(columns[0])
plt.ylabel("Количество")
plt.tight_layout()
plt.show()

# 5. Собираем boxplot’ы по классам для каждого количественного признака
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
for ax, col in zip(axes.flatten(), numeric_cols):
    sns.boxplot(x=columns[0], y=col, data=data, ax=ax, palette='Set3')
    ax.set_title(col)
# удаляем лишние подплоты
for ax in axes.flatten()[n:]:
    fig.delaxes(ax)
plt.tight_layout()
plt.show()

# 6. Корреляционная тепловая карта (отдельно)
plt.figure(figsize=(12, 10))
corr = data[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Тепловая карта корреляции количественных признаков")
plt.tight_layout()
plt.show()

# 7. Pairplot по первым пяти признакам (отдельно)
sns.pairplot(data, hue=columns[0], vars=numeric_cols[:5], palette='tab10')
plt.suptitle("Pairplot первых 5 признаков", y=1.02)
plt.show()
