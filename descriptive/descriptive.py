import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Загрузка данных
df = pd.read_csv('employee.csv')

# Filling in missing values
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=[object]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Function for replacing abnormal values
def replace_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = series.median()
    return series.apply(lambda x: median if x < lower_bound or x > upper_bound else x)

# Replacing anomalous values in numeric columns
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'ID':  # Исключаем столбец ID
        df[col] = replace_outliers(df[col])

# Основная информация о данных
print("Основная информация о данных:")
print(df.info())

# Описание данных
print("\nОписание данных:")
print(df.describe())

# Проверка пропущенных значений
print("\nПропущенные значения в данных:")
print(df.isnull().sum())

# Распределение числовых данных
numerical_columns = df.select_dtypes(include=[np.number]).columns
numerical_columns = numerical_columns.drop('ID')  # Исключаем столбец ID
df[numerical_columns].hist(figsize=(10, 10), bins=30, edgecolor='k')
plt.suptitle('Распределение числовых данных')
plt.show()

# Боксплоты для числовых данных
num_columns_count = len(numerical_columns)
layout_rows = (num_columns_count // 2) + (num_columns_count % 2)
df[numerical_columns].plot(kind='box', subplots=True, layout=(layout_rows, 2), figsize=(12, layout_rows * 5), title='Boxplot числовых данных')
plt.tight_layout()
plt.show()

# Распределение категориальных данных
categorical_columns = df.select_dtypes(include=[object]).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Распределение значений для {col}')
    plt.show()

# Корреляция между числовыми признаками
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
plt.imshow(correlation_matrix, cmap='coolwarm', interpolation='none')
plt.colorbar()
plt.xticks(range(len(correlation_matrix)), correlation_matrix.columns, rotation=90)
plt.yticks(range(len(correlation_matrix)), correlation_matrix.columns)
plt.title('Корреляционная матрица числовых признаков')
plt.show()

# Парные графики для числовых данных
pd.plotting.scatter_matrix(df[numerical_columns], figsize=(12, 12), diagonal='kde')
plt.suptitle('Парные графики числовых данных')
plt.show()
