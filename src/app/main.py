from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from recommender import Recommender
import joblib
import pandas as pd
import numpy as np

# Инициализация FastAPI приложения
app = FastAPI()

# Модели данных для запроса
class RecommendationRequest(BaseModel):
    user_id: int
    num_recommendations: int = 5

# Пути к файлам модели и данных
MODEL_PATH = "catboost_recommender.pkl"  # Укажите правильный путь к файлу модели
OVERVIEW_PATH = "../data/overview_vectors.csv"  # Укажите правильный путь к векторным данным фильмов
RATINGS_PATH = "../data/rating.csv"  # Укажите правильный путь к файлу с рейтингами

# Создание экземпляра recommender с нужными путями к данным
recommender = Recommender(MODEL_PATH, OVERVIEW_PATH)

@app.get("/")
def read_root():
    return {"message": "Welcome to the movie recommender API!"}

@app.post("/recommendations/")
async def get_recommendations(request: RecommendationRequest):
    try:
        # Получаем рекомендации для пользователя
        recommendations = recommender.get_recommendations(request.user_id, request.num_recommendations)
        return recommendations
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))