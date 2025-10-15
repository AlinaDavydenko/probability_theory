import pandas as pd
import numpy as np
from metrics import y_pred
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Загружаем данные из CSV файла
df = pd.read_csv('datasets/dataset_diseases.csv')

# Первые 10 строк данных
# print(df.head(10))

# Предобработка данных, добавление числовых значений
le_test = LabelEncoder()
le_age = LabelEncoder()
le_status = LabelEncoder()

df['Test_encoded'] = le_test.fit_transform(df['Test'])
df['Age_Group_encoded'] = le_age.fit_transform(df['Age_Group'])
df['Status_encoded'] = le_status.fit_transform(df['Status'])

# Разделение данных на признаки и целевые переменные
X = df[['Test_encoded', 'Age_Group_encoded']]  # Признаки
y = df['Status_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)  # stratify=y
# тестовая часть 30% от всех данных

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_test shape: {y_test.shape}")

# Обучение модели Наивного Байса
model = CategoricalNB()
model.fit(X_train, y_train)

# Предсказания модели
y_pred = model.predict(X_test)

print(f"y_pred shape: {y_pred.shape}")
print(f"y_test shape: {y_test.shape}")

# Совпадение размеров % (т.к. была ошибка)
if y_pred.shape != y_test.shape:
    print(f"Ошибка: размеры не совпадают \n y_pred: {y_pred.shape}, y_test: {y_test.shape}")
    # Используем меньший размер для сравнения
    min_size = min(len(y_pred), len(y_test))
    y_pred = y_pred[:min_size]
    y_test = y_test[:min_size]
    print(f"Обрезано до: y_pred: {y_pred.shape}, y_test: {y_test.shape}")

# Оценка качество
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

# Декодируем обратно для читаемого отчета
y_test_decoded = le_status.inverse_transform(y_test)
y_pred_decoded = le_status.inverse_transform(y_pred)

print("Отчет о классификации:")
print(classification_report(y_test_decoded, y_pred_decoded))

print("Матрица ошибок:")
print(confusion_matrix(y_test_decoded, y_pred_decoded))
