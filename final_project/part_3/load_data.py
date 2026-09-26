from __future__ import annotations

import argparse

import pandas as pd

from database import postgres_connection


DEFAULT_FEED_LIMIT = 5_000_000
MAX_FEED_LIMIT = 10_000_000


def load_users(connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT user_id, age, gender, country, city, exp_group, os, source
        FROM public.user_data
        """,
        connection,
    )


def load_posts(connection) -> pd.DataFrame:
    return pd.read_sql(
        """
        SELECT post_id, text, topic
        FROM public.post_text_df
        """,
        connection,
    )


def load_feed(connection, limit: int = DEFAULT_FEED_LIMIT) -> pd.DataFrame:
    if not 1 <= limit <= MAX_FEED_LIMIT:
        raise ValueError(f"limit должен быть от 1 до {MAX_FEED_LIMIT}")

    return pd.read_sql(
        f"""
        SELECT timestamp, user_id, post_id, action, target
        FROM public.feed_data
        WHERE action = 'view'
        LIMIT {limit}
        """,
        connection,
    )


def load_all_data(feed_limit: int = DEFAULT_FEED_LIMIT):
    connection = postgres_connection()
    try:
        users = load_users(connection)
        posts = load_posts(connection)
        feed = load_feed(connection, limit=feed_limit)
    finally:
        connection.close()

    return users, posts, feed


def validate_loaded_data(users, posts, feed) -> None:
    if not {"user_id", "age", "gender", "country", "city", "exp_group", "os", "source"}.issubset(users.columns):
        raise ValueError("В user_data отсутствуют обязательные столбцы")
    if not {"post_id", "text", "topic"}.issubset(posts.columns):
        raise ValueError("В post_text_df отсутствуют обязательные столбцы")
    if not {"timestamp", "user_id", "post_id", "action", "target"}.issubset(feed.columns):
        raise ValueError("В feed_data отсутствуют обязательные столбцы")
    if len(feed) > MAX_FEED_LIMIT:
        raise ValueError("Выгружено больше 10 миллионов взаимодействий")
    if feed["target"].isna().any():
        raise ValueError("В target есть пропуски")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feed-limit", type=int, default=DEFAULT_FEED_LIMIT)
    args = parser.parse_args()

    users, posts, feed = load_all_data(args.feed_limit)
    validate_loaded_data(users, posts, feed)

    print("user_data:", users.shape)
    print("post_text_df:", posts.shape)
    print("feed_data:", feed.shape)
    print(feed["target"].value_counts(dropna=False))


if __name__ == "__main__":
    main()
