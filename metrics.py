import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import recall_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score

# Загрузка данных (замените на ваш путь к файлу)
spam_data = pd.read_csv('datasets/spam_or_not_spam.csv')

# 1. Предобработка - удаление пустых строк и строк только из пробелов
spam_data_clean = spam_data[spam_data['email'].str.strip().astype(bool)]

print(f"Количество строк до очистки: {len(spam_data)}")
print(f"Количество строк после очистки: {len(spam_data_clean)}")

# 2. Применение CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(spam_data_clean['email'])

# 3. Получение информации о матрице
num_features = X.shape[1]
num_documents = X.shape[0]

print(f"\nРезультаты векторизации:")
print(f"Количество документов: {num_documents}")
print(f"Количество признаков (уникальных слов): {num_features}")
print(f"Размерность матрицы: {X.shape}")

# 4. Дополнительная информация
print(f"\nТип матрицы: {type(X)}")
print(f"Примеры признаков (первые 20): {vectorizer.get_feature_names_out()[:20]}")

# 5. Проверка разреженности матрицы
total_elements = X.shape[0] * X.shape[1]
non_zero_elements = X.nnz
sparsity = (1 - non_zero_elements / total_elements) * 100

print(f"\nАнализ разреженности:")
print(f"Всего элементов в матрице: {total_elements}")
print(f"Ненулевых элементов: {non_zero_elements}")
print(f"Разреженность: {sparsity:.2f}%")


# Сначала убедимся, что в целевой переменной нет NaN
print(f"NaN values in target: {spam_data_clean['label'].isna().sum()}")

# Если есть NaN, удалим их
spam_data_final = spam_data_clean.dropna(subset=['label'])

# Пересоздадим матрицу признаков для очищенных данных
vectorizer = CountVectorizer()
X_clean = vectorizer.fit_transform(spam_data_final['email'])
y_clean = spam_data_final['label']

print(f"Данные после очистки от NaN:")
print(f"Размер X: {X_clean.shape}")
print(f"Размер y: {len(y_clean)}")

# Теперь разделяем очищенные данные
X_train, X_test, y_train, y_test = train_test_split(
    X_clean,
    y_clean,
    train_size=0.75,
    test_size=0.25,
    stratify=y_clean,  # стратифицированное разбиение
    random_state=42
)

# Вычисляем среднее значение целевой переменной
mean_target = y_train.mean()

print(f"\nРезультаты разбиения:")
print(f"Размер обучающей выборки: {X_train.shape[0]}")
print(f"Размер тестовой выборки: {X_test.shape[0]}")
print(f"Среднее значение целевой переменной (y_train): {mean_target:.3f}")

# Дополнительная проверка распределения
print(f"\nРаспределение в обучающей выборке:")
print(y_train.value_counts(normalize=True))
print(f"\nРаспределение в тестовой выборке:")
print(y_test.value_counts(normalize=True))

# Обучаем модель с alpha=0.01
model = MultinomialNB(alpha=0.01)
model.fit(X_train, y_train)

# Предсказания на тестовой выборке
y_pred = model.predict(X_test)

# Вычисляем метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Метрики качества на тестовой выборке:")
print(f"Accuracy: {accuracy:.3f}")
print(f"Precision: {precision:.3f}")

# Дополнительные метрики для полноты картины

recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Recall: {recall:.3f}")
print(f"F1-score: {f1:.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))