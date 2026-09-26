import os
import pickle
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI
from loguru import logger

from database import postgres_connection
from schema import PostGet


def load_sql(query: str, dtypes: Dict[str, Any] | None = None) -> pd.DataFrame:
    conn = postgres_connection()
    try:
        return pd.read_sql(query, conn, dtype=dtypes)
    finally:
        conn.close()


def load_model(model_path: str = "model.pkl"):
    if os.environ.get("IS_LMS", "0") == "1":
        model_path = os.environ["MODEL_PATH"]

    with open(model_path, "rb") as file:
        return pickle.load(file)


CAT_FEATURES = ["gender", "country", "city", "exp_group", "os", "source", "topic"]
EMB_COLS = [f"emb_{i}" for i in range(15)]
FEATURES = (
    CAT_FEATURES
    + ["age", "text_length", "word_count", "unique_word_count", "hour", "dayofweek", "month"]
    + EMB_COLS
)

logger.info("Инициализация сервиса...")

app = FastAPI()
model = load_model()

user_features = load_sql(
    """
    SELECT user_id, age, gender, country, city, exp_group, os, source
    FROM public.user_data
    """
)
for column in ["gender", "country", "city", "exp_group", "os", "source"]:
    user_features[column] = user_features[column].astype(str)
user_features = user_features.set_index("user_id")

post_features = load_sql(
    """
    SELECT *
    FROM nina14726_post_features_dl
    """
)
post_features["topic"] = post_features["topic"].astype(str)
post_features = post_features.set_index("post_id")

posts_text = load_sql(
    """
    SELECT post_id, text, topic
    FROM public.post_text_df
    """
).set_index("post_id")

logger.success("Сервис успешно инициализирован")


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(user_id: int, dt: datetime, limit: int = 10) -> List[PostGet]:
    try:
        user = user_features.loc[user_id]
    except KeyError:
        return []

    X = post_features.copy()
    X["gender"] = user["gender"]
    X["country"] = user["country"]
    X["city"] = user["city"]
    X["exp_group"] = user["exp_group"]
    X["os"] = user["os"]
    X["source"] = user["source"]
    X["age"] = user["age"]
    X["hour"] = dt.hour
    X["dayofweek"] = dt.weekday()
    X["month"] = dt.month
    X = X[FEATURES]

    proba = model.predict_proba(X)[:, 1]
    top_idx = np.argsort(proba)[::-1][:limit]
    top_post_ids = post_features.index.to_numpy()[top_idx]

    result = []
    for post_id in top_post_ids:
        row = posts_text.loc[post_id]
        result.append(
            PostGet(
                id=int(post_id),
                text=str(row["text"]),
                topic=None if pd.isna(row["topic"]) else str(row["topic"]),
            )
        )

    return result
