import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import randint

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

# Преобразование категориальных данных в числовые с помощью OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)
encoded_depts = encoder.fit_transform(df[['Department']])
encoded_dept_df = pd.DataFrame(encoded_depts, columns=encoder.get_feature_names_out(['Department']))

# Объединяем закодированные данные с основным DataFrame
df = pd.concat([df, encoded_dept_df], axis=1)

# Удаляем исходный столбец Department
df.drop('Department', axis=1, inplace=True)

# Подготовка данных для модели
X = df[['Age'] + list(encoded_dept_df.columns)]
y = df['Salary']

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание и обучение модели случайного леса с подбором гиперпараметров
rf = RandomForestRegressor(random_state=42)

# Определение параметров для RandomizedSearchCV
param_distributions = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 10),
    'min_samples_leaf': randint(1, 4)
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_distributions, n_iter=100, cv=5, n_jobs=-1, random_state=42, scoring='r2')
random_search.fit(X_train, y_train)

# Лучшая модель
best_rf = random_search.best_estimator_

# Прогнозирование на тестовой выборке
y_pred = best_rf.predict(X_test)

# Оценка модели
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error (MSE):", mse)
print("R-squared (R2):", r2)

# Визуализация прогнозов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, edgecolors=(0, 0, 0))
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=4)
plt.xlabel('Actual Salary')
plt.ylabel('Predicted Salary')
plt.title('Actual vs Predicted Salary')
plt.grid(True)
plt.show()

# Важность признаков
feature_importance = pd.DataFrame(best_rf.feature_importances_, index=X.columns, columns=['Importance'])
print("\nFeature Importance:")
print(feature_importance)
