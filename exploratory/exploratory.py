import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Загрузка данных
df = pd.read_csv('employee.csv')
# Заполнение пропущенных значений
for col in df.select_dtypes(include=[np.number]).columns:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include=[object]).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Функция для замены аномальных значений
def replace_outliers(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    median = series.median()
    return series.apply(lambda x: median if x < lower_bound or x > upper_bound else x)

# Замена аномальных значений в числовых столбцах
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

# Анализ распределения числовых данных
numerical_columns = df.select_dtypes(include=[np.number]).columns

# Гистограммы числовых данных
df[numerical_columns].hist(figsize=(10, 10), bins=30, edgecolor='k')
plt.suptitle('Распределение числовых данных')
plt.show()

# Боксплоты числовых данных
num_columns_count = len(numerical_columns)
layout_rows = (num_columns_count // 2) + (num_columns_count % 2)
df[numerical_columns].plot(kind='box', subplots=True, layout=(layout_rows, 2), figsize=(12, layout_rows * 5), title='Boxplot числовых данных')
plt.tight_layout()
plt.show()

# Анализ распределения категориальных данных
categorical_columns = df.select_dtypes(include=[object]).columns
for col in categorical_columns:
    plt.figure(figsize=(8, 4))
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Распределение значений для {col}')
    plt.show()

# Корреляционный анализ числовых данных
correlation_matrix = df[numerical_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Корреляционная матрица числовых признаков')
plt.show()

# Парные графики числовых данных
sns.pairplot(df[numerical_columns])
plt.suptitle('Парные графики числовых данных')
plt.show()

# Анализ временных данных (если применимо)
if 'Joining_Date' in df.columns:
    df['Joining_Date'] = pd.to_datetime(df['Joining_Date'])
    df.set_index('Joining_Date', inplace=True)
    df['Year'] = df.index.year
    df['Month'] = df.index.month

    # Анализ по годам
    plt.figure(figsize=(10, 5))
    df['Year'].value_counts().sort_index().plot(kind='bar')
    plt.title('Количество присоединений по годам')
    plt.show()

    # Анализ по месяцам
    plt.figure(figsize=(10, 5))
    df['Month'].value_counts().sort_index().plot(kind='bar')
    plt.title('Количество присоединений по месяцам')
    plt.show()

# Анализ распределения зарплаты по отделам
if 'Salary' in df.columns and 'Department' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Department', y='Salary', data=df)
    plt.title('Распределение зарплаты по отделам')
    plt.show()
