from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI
from loguru import logger

from database import postgres_connection
from schema import PostGet


def load_sql(query: str, dtypes: Dict[str, Any] | None = None) -> pd.DataFrame:
    """Выполняет SQL-запрос и возвращает результат в DataFrame."""
    conn = postgres_connection()

    try:
        df = pd.read_sql(query, conn, dtype=dtypes)
    except Exception as exc:
        raise RuntimeError(
            f"Ошибка при выполнении SQL-запроса: {exc}\nЗапрос: {query}"
        ) from exc
    finally:
        conn.close()

    return df


logger.info("Инициализация сервиса...")

app = FastAPI()

logger.info("Загружаем список постов...")
df_posts_sorted = load_sql(
    """
    SELECT post_id AS id, text, topic
    FROM public.post_text_df
    ORDER BY post_id
    """,
    dtypes={"id": "int64"},
)
logger.success("Список постов успешно загружен.")
logger.success("Сервис успешно инициализирован")


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(
    user_id: int,
    dt: datetime,
    limit: int = 10,
) -> List[PostGet]:
    """Возвращает первые limit постов той же чётности, что и user_id."""
    if user_id % 2 == 0:
        top_posts = df_posts_sorted[df_posts_sorted["id"] % 2 == 0].head(limit)
    else:
        top_posts = df_posts_sorted[df_posts_sorted["id"] % 2 != 0].head(limit)

    recs = [
        PostGet(
            id=int(row.id),
            text=str(row.text),
            topic=None if pd.isna(row.topic) else str(row.topic),
        )
        for row in top_posts.itertuples(index=False)
    ]

    return recs
