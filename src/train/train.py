import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.preprocessing import MultiLabelBinarizer
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity

print("Загрузка стоп-слов")
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    words = [word for word in words if word.lower() not in stop_words]
    return " ".join(words)

print("Загрузка данных")
movie = pd.read_csv("../resources/movie.csv")
ratings = pd.read_csv("../resources/rating.csv").sample(frac=0.1, random_state=42)
merged_movies = pd.read_csv("../resources/merged_movies.csv").sample(frac=0.1, random_state=42)

print("Удаление дубликатов")
ratings = ratings.drop_duplicates()
merged_movies = merged_movies.drop_duplicates()

print("Разбиение жанров на отдельные метки")
merged_movies['genres'] = merged_movies['genres'].apply(lambda x: x.split('|'))

print("One-Hot Encoding для жанров")
mlb = MultiLabelBinarizer()
genres_encoded = mlb.fit_transform(merged_movies['genres'])
genres_df = pd.DataFrame(genres_encoded, columns=mlb.classes_)
merged_movies = pd.concat([merged_movies, genres_df], axis=1).drop(columns=['genres'])

print("Подготовка описаний для обучения Word2Vec")
merged_movies['overview'] = merged_movies['overview'].fillna('').apply(remove_stopwords)
merged_movies['overview_tokens'] = merged_movies['overview'].fillna('').apply(lambda x: x.split())

print("Обучение модели Word2Vec")
word2vec_model = Word2Vec(sentences=merged_movies['overview_tokens'], vector_size=100, window=5, min_count=1, workers=4)

print("Функция для усреднения векторов слов")
def average_word2vec(tokens, model, vector_size=100):
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if len(vectors) > 0:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(vector_size)

merged_movies['overview_vector'] = merged_movies['overview_tokens'].apply(lambda x: average_word2vec(x, word2vec_model))

print("Преобразование списка векторов в DataFrame")
overview_vectors = np.vstack(merged_movies['overview_vector'].values)
overview_df = pd.DataFrame(overview_vectors, columns=[f'feature_{i}' for i in range(overview_vectors.shape[1])])
overview_df['movieId'] = merged_movies['movieId']

print("Сохраняем векторные представления фильмов")
overview_df.to_csv("../resources/overview_vectors.csv", index=False)

print("Объединяем ratings с данными о фильмах")
data = pd.merge(ratings, overview_df, on='movieId', how='inner')
data['target'] = data['rating'].apply(lambda x: 1 if x >= 4 else 0)

print("Вычисление пользовательских предпочтений")
user_preferences = ratings.groupby('userId').agg(
    avg_rating=('rating', 'mean'),
    num_ratings=('rating', 'count')
)

print("Добавляем пользовательские метрики")

data = pd.merge(data, user_preferences, on='userId', how='left')

print("Группируем понравившиеся фильмы для каждого пользователя")
vector_columns = [f'feature_{i}' for i in range(overview_vectors.shape[1])]

print("Группируем понравившиеся фильмы для каждого пользователя")
liked_movies = data[data['target'] == 1].groupby('userId')[vector_columns].mean()

print("Сбрасываем индекс для удобства")
liked_movies = liked_movies.reset_index()

# Создаем колонку liked_vector, представляющую усредненные векторы
liked_movies['liked_vector'] = liked_movies[vector_columns].values.tolist()

# Сохраняем только userId и liked_vector
liked_movies = liked_movies[['userId', 'liked_vector']]

print("Присоединяем усреднённые векторы понравившихся фильмов")
data = pd.merge(data, liked_movies, on='userId', how='left')
data['similarity_to_user'] = data.apply(
    lambda row: cosine_similarity(
        [row[[f'feature_{i}' for i in range(overview_vectors.shape[1])]].values],
        [row['liked_vector']]
    ).flatten()[0] if isinstance(row['liked_vector'], list) else 0,
    axis=1
)

# Определение признаков и целевой переменной
features = ['avg_rating', 'num_ratings', 'similarity_to_user'] + [f'feature_{i}' for i in range(overview_vectors.shape[1])]
X = data[features]
y = data['target']

print("Разделение на обучающую и тестовую выборки")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("добрались до создания и обучения модели")
model = CatBoostClassifier(iterations=500, learning_rate=0.1, depth=8, task_type="GPU", logging_level="Verbose")
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=10)

# Сохраняем модель на диск
joblib.dump(model, '../app/catboost_recommender.pkl')

# Метрики точности и F1
y_pred = (model.predict_proba(X_test)[:, 1] >= 0.50).astype(int)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(f"F1 Score: {f1:.2f}")


# def recommend_movies(user_id, num_recommendations):
#     print("Этап 1: Проверка наличия пользователя в данных...")
#     if user_id not in data['userId'].values:
#         print(f"Пользователь с ID {user_id} не найден.")
#         return []
#
#     print("Этап 2: Фильтрация фильмов, которые пользователь уже оценил...")
#     user_rated_movies = data[data['userId'] == user_id]['movieId'].unique()
#     unrated_movies = merged_movies[~merged_movies['movieId'].isin(user_rated_movies)]
#     print(f"Количество фильмов для рекомендации: {len(unrated_movies)}")
#
#     print("Этап 3: Добавление векторов признаков для фильмов...")
#     unrated_movies_with_features = pd.merge(
#         unrated_movies,
#         overview_df,
#         on='movieId',
#         how='inner'
#     )
#     print(f"Количество фильмов после объединения с признаками: {len(unrated_movies_with_features)}")
#     if unrated_movies_with_features['movieId'].isna().any():
#         print("Предупреждение: Найдены NaN в столбце movieId.")
#         unrated_movies_with_features = unrated_movies_with_features.dropna(subset=['movieId'])
#
#     print("Этап 4: Получение усреднённого вектора предпочтений пользователя...")
#     user_liked_vector_row = liked_movies[liked_movies['userId'] == user_id]
#     if user_liked_vector_row.empty:
#         print(f"У пользователя с ID {user_id} недостаточно данных для рекомендаций.")
#         return []
#
#     user_liked_vector = np.array(user_liked_vector_row['liked_vector'].values[0]).reshape(1, -1)
#     print("Вектор предпочтений пользователя успешно получен.")
#
#     print("Этап 5: Вычисление сходства для всех фильмов...")
#     movie_vectors = np.vstack(unrated_movies_with_features[[f'feature_{i}' for i in range(overview_vectors.shape[1])]].values)
#     similarities = cosine_similarity(movie_vectors, user_liked_vector).flatten()
#     unrated_movies_with_features['similarity_to_user'] = similarities
#     print("Сходство рассчитано.")
#
#     print("Этап 6: Добавление метрик пользователя...")
#     avg_rating = data[data['userId'] == user_id]['avg_rating'].mean()
#     num_ratings = data[data['userId'] == user_id]['num_ratings'].mean()
#     unrated_movies_with_features['avg_rating'] = avg_rating
#     unrated_movies_with_features['num_ratings'] = num_ratings
#     print("Метрики пользователя добавлены.")
#
#     print("Этап 7: Формирование признаков для предсказания...")
#     features_for_prediction = ['avg_rating', 'num_ratings', 'similarity_to_user'] + \
#                               [f'feature_{i}' for i in range(overview_vectors.shape[1])]
#     X_unrated = unrated_movies_with_features[features_for_prediction]
#     print("Признаки сформированы.")
#     print(X_unrated.describe())  # Проверяем на одинаковость
#
#     print("Этап 8: Прогноз вероятностей для фильмов...")
#     probabilities = model.predict_proba(X_unrated)[:, 1]
#     unrated_movies_with_features['predicted_probability'] = probabilities
#     print("Прогноз вероятностей завершён.")
#     print(f"Минимальная вероятность: {probabilities.min()}, Максимальная вероятность: {probabilities.max()}")
#
#     print("Этап 9: Сортировка фильмов по вероятности...")
#     top_indices = np.argpartition(probabilities, -num_recommendations)[-num_recommendations:]
#     recommendations = unrated_movies_with_features.iloc[top_indices].sort_values('predicted_probability', ascending=False)
#     print("Сортировка завершена.")
#     print(recommendations[['movieId', 'predicted_probability']].head())
#
#     print("Этап 10: Формирование результата...")
#     result = recommendations[['movieId', 'predicted_probability']].values.tolist()
#     print("Рекомендации сформированы.")
#
#     return result

# user_id = 49851
# num_recommendations = 5
# recommendations = recommend_movies(user_id, num_recommendations)
# print(f"Рекомендации для пользователя {user_id}: {recommendations}")
