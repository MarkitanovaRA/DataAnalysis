import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
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

# t-тест для проверки гипотезы о среднем значении возраста (Age)
age_mean_hypothesis = 40  # Гипотетическое среднее значение
t_stat, p_value = stats.ttest_1samp(df['Age'], age_mean_hypothesis)
print("\nT-test для среднего значения возраста:")
print(f"t-statistic: {t_stat}, p-value: {p_value}")

# Корреляционный анализ между возрастом (Age) и зарплатой (Salary)
correlation, p_value = stats.pearsonr(df['Age'], df['Salary'])
print("\nКорреляция между возрастом и зарплатой:")
print(f"Корреляция: {correlation}, p-value: {p_value}")

# Хи-квадрат тест для анализа зависимости между отделом (Department) и зарплатой (Salary)
# Преобразуем Salary в категориальную переменную
df['Salary_Category'] = pd.qcut(df['Salary'], q=3, labels=['Low', 'Medium', 'High'])
contingency_table = pd.crosstab(df['Department'], df['Salary_Category'])
chi2, p, dof, ex = stats.chi2_contingency(contingency_table)
print("\nХи-квадрат тест для зависимости между отделом и категорией зарплаты:")
print(f"chi2: {chi2}, p-value: {p}")

# Корреляционная матрица для числовых данных (исключая ID)
numerical_columns = df.select_dtypes(include=[np.number]).columns
numerical_columns = [col for col in numerical_columns if col != 'ID']  # Исключаем столбец ID
correlation_matrix = df[numerical_columns].corr()

# График корреляционной матрицы
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Корреляционная матрица')
plt.show()

# График распределения возраста
sns.histplot(df['Age'], kde=True)
plt.title('Распределение возраста')
plt.show()

# График распределения зарплаты
sns.histplot(df['Salary'], kde=True)
plt.title('Распределение зарплаты')
plt.show()

# График boxplot для зарплаты по отделам
plt.figure(figsize=(10, 6))
sns.boxplot(x='Department', y='Salary', data=df)
plt.title('Boxplot зарплаты по отделам')
plt.show()
