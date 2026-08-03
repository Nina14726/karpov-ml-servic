from __future__ import annotations

import argparse

import pandas as pd

from database import postgres_connection


DEFAULT_FEED_LIMIT = 5_000_000
MAX_FEED_LIMIT = 10_000_000


def load_users(connection) -> pd.DataFrame:
    """Загружает профили всех пользователей."""
    return pd.read_sql(
        """
        SELECT user_id, age, gender, country, city, exp_group, os, source
        FROM public.user_data
        """,
        connection,
    )


def load_posts(connection) -> pd.DataFrame:
    """Загружает тексты и тематики всех постов."""
    return pd.read_sql(
        """
        SELECT id, text, topic
        FROM public.post_text_df
        """,
        connection,
    )


def load_feed(connection, limit: int = DEFAULT_FEED_LIMIT) -> pd.DataFrame:
    """Загружает просмотры с целевой переменной для обучения модели."""
    if not 1 <= limit <= MAX_FEED_LIMIT:
        raise ValueError(f"limit должен быть от 1 до {MAX_FEED_LIMIT}")

    query = f"""
        SELECT timestamp, user_id, post_id, action, target
        FROM public.feed_data
        WHERE action = 'view'
        LIMIT {limit}
    """
    return pd.read_sql(query, connection)


def load_all_data(
    feed_limit: int = DEFAULT_FEED_LIMIT,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает датафреймы пользователей, постов и взаимодействий."""
    connection = postgres_connection()
    try:
        users = load_users(connection)
        posts = load_posts(connection)
        feed = load_feed(connection, limit=feed_limit)
    finally:
        connection.close()

    return users, posts, feed


def validate_loaded_data(
    users: pd.DataFrame,
    posts: pd.DataFrame,
    feed: pd.DataFrame,
) -> None:
    """Проверяет обязательные столбцы и основные ограничения задания."""
    expected_users = {
        "user_id",
        "age",
        "gender",
        "country",
        "city",
        "exp_group",
        "os",
        "source",
    }
    expected_posts = {"id", "text", "topic"}
    expected_feed = {"timestamp", "user_id", "post_id", "action", "target"}

    if not expected_users.issubset(users.columns):
        raise ValueError("В user_data отсутствуют обязательные столбцы")
    if not expected_posts.issubset(posts.columns):
        raise ValueError("В post_text_df отсутствуют обязательные столбцы")
    if not expected_feed.issubset(feed.columns):
        raise ValueError("В feed_data отсутствуют обязательные столбцы")
    if len(feed) > MAX_FEED_LIMIT:
        raise ValueError("Выгружено больше 10 миллионов взаимодействий")
    if not feed.empty and not feed["action"].eq("view").all():
        raise ValueError("В обучающей выборке должны находиться только просмотры")
    if feed["target"].isna().any():
        raise ValueError("В target обнаружены пропущенные значения")


def main() -> None:
    parser = argparse.ArgumentParser(description="Выгрузка данных финального проекта")
    parser.add_argument(
        "--feed-limit",
        type=int,
        default=DEFAULT_FEED_LIMIT,
        help="Количество строк public.feed_data (не более 10 000 000)",
    )
    args = parser.parse_args()

    users, posts, feed = load_all_data(feed_limit=args.feed_limit)
    validate_loaded_data(users, posts, feed)

    print("user_data:", users.shape)
    print(users.head())
    print("\npost_text_df:", posts.shape)
    print(posts.head())
    print("\nfeed_data:", feed.shape)
    print(feed.head())
    print("\ntarget distribution:")
    print(feed["target"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
