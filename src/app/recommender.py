import pandas as pd
import numpy as np
import psycopg2
from sklearn.metrics.pairwise import cosine_similarity
import joblib


class Recommender:
    def __init__(self, model_path: str, overview_path: str):
        self.connection = psycopg2.connect(
            host="localhost",
            port="5432",
            database="mldata",
            user="anton",
            password="1234"
        )
        # Загрузка модели
        self.model = joblib.load(model_path)

        # Загрузка векторных представлений фильмов
        self.overview_df = pd.read_csv(overview_path)

        # Выполнение SQL-запроса для загрузки оценок
        ratings_query = "SELECT userId, movieId, rating, \"timestamp\" FROM ratings"
        self.ratings = pd.read_sql_query(ratings_query, self.connection)

        movies_query = "SELECT movieId,title,genres FROM movies"
        self.movies = pd.read_sql_query(movies_query, self.connection)

        self.data = self.ratings.merge(self.movies, on="movieId")

        # Вычисление пользовательских предпочтений
        self.user_preferences = self.ratings.groupby('userId').agg(
            avg_rating=('rating', 'mean'),
            num_ratings=('rating', 'count')
        ).reset_index()

        # Объединение понравившихся фильмов с векторными представлениями
        self.vector_columns = [col for col in self.overview_df.columns if col.startswith("feature_")]
        self.liked_movies = self._compute_user_liked_vectors()

    def _compute_user_liked_vectors(self):
        """
        Вычисляет усреднённые векторы понравившихся фильмов для каждого пользователя.
        """
        # Определяем таргет (положительные отзывы - >= 4)
        self.ratings['target'] = self.ratings['rating'].apply(lambda x: 1 if x >= 4 else 0)

        # Группируем понравившиеся фильмы для каждого пользователя
        liked_movies = self.ratings[self.ratings['target'] == 1].merge(
            self.overview_df, on='movieId', how='inner'
        ).groupby('userId')[self.vector_columns].mean().reset_index()

        # Создаём колонку liked_vector
        liked_movies['liked_vector'] = liked_movies[self.vector_columns].values.tolist()

        return liked_movies[['userId', 'liked_vector']]

    def get_recommendations(self, user_id: int, num_recommendations: int = 5):
        """
        Рекомендует фильмы для пользователя на основе модели CatBoost и сходства с векторами.
        """
        # Проверка наличия пользователя
        if user_id not in self.ratings['userId'].unique():
            return self.cold_search(num_recommendations)

        # Фильтрация фильмов, которые пользователь уже оценил
        user_rated_movies = self.ratings[self.ratings['userId'] == user_id]['movieId'].unique()
        unrated_movies = self.overview_df[~self.overview_df['movieId'].isin(user_rated_movies)]

        # Получение вектора предпочтений пользователя
        user_liked_vector_row = self.liked_movies[self.liked_movies['userId'] == user_id]
        if user_liked_vector_row.empty:
            return self.cold_search(num_recommendations)

        user_liked_vector = np.array(user_liked_vector_row['liked_vector'].values[0]).reshape(1, -1)

        # Вычисление сходства с пользователем
        movie_vectors = np.vstack(unrated_movies[self.vector_columns].values)
        similarities = cosine_similarity(movie_vectors, user_liked_vector).flatten()
        unrated_movies['similarity_to_user'] = similarities

        # Добавление пользовательских метрик
        user_preferences_row = self.user_preferences[self.user_preferences['userId'] == user_id]
        avg_rating = user_preferences_row['avg_rating'].iloc[0] if not user_preferences_row.empty else 0
        num_ratings = user_preferences_row['num_ratings'].iloc[0] if not user_preferences_row.empty else 0
        unrated_movies['avg_rating'] = avg_rating
        unrated_movies['num_ratings'] = num_ratings

        # Формирование признаков для предсказания
        features_for_prediction = ['avg_rating', 'num_ratings', 'similarity_to_user'] + self.vector_columns
        X_unrated = unrated_movies[features_for_prediction]

        # Прогноз вероятностей для фильмов
        probabilities = self.model.predict_proba(X_unrated)[:, 1]
        unrated_movies['predicted_probability'] = probabilities

        # Сортировка фильмов по вероятности
        recommendations = (
            unrated_movies.nlargest(num_recommendations, 'predicted_probability')[['movieId', 'predicted_probability']]
        )

        # Формирование результата
        return recommendations.to_dict(orient='records')

    def cold_search(self, num_recommendations):
        # Calculate the average rating and number of ratings for each movie
        movie_stats = self.data.groupby("title").agg(
            avg_rating=("rating", "mean"),  # Средний рейтинг
            num_ratings=("rating", "count")  # Количество оценок
        ).reset_index()

        # Filter movies with average rating >= 4.00
        high_rated_movies = movie_stats[movie_stats["avg_rating"] >= 4.00]

        # Sort movies by number of ratings in descending order
        sorted_movies = high_rated_movies.sort_values(by="num_ratings", ascending=False)

        # Get the top N movies
        top_movies = sorted_movies.head(num_recommendations)

        # Return only the list of movie titles
        return top_movies["title"].tolist()