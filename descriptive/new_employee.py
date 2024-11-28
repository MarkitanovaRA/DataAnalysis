import pandas as pd
import numpy as np

# Создание DataFrame с примерными данными с пропусками и аномалиями
data = {
    'ID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eva', 'Frank', 'Grace', 'Hannah', 'Ian', 'Judy', 'Karl', 'Laura'],
    'Age': [25, 30, np.nan, 40, 45, 50, -5, 60, 65, 70, 200, 55],
    'Salary': [50000, 60000, 70000, 80000, np.nan, 100000, 110000, 120000, 130000, -1000, 5000000, 140000],
    'Department': ['HR', 'Engineering', 'Finance', 'Marketing', 'Engineering', 'Finance', np.nan, 'Marketing', 'Engineering', 'HR', 'HR', 'Engineering'],
    'Joining_Date': ['2018-01-15', '2017-03-22', '2016-07-11', '2019-05-30', '2015-11-20', '2014-09-25', '2013-08-19', '2020-02-15', '2012-06-12', '2011-05-01', '2023-15-10', '2010-01-35']
}

df = pd.DataFrame(data)

# Сохранение DataFrame в CSV файл
df.to_csv('your_dataset_with_anomalies.csv', index=False)
