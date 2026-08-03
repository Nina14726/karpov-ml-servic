import os

import psycopg2


def postgres_connection():
    """Устанавливает и возвращает соединение с PostgreSQL."""
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "postgres.lab.karpov.courses"),
            port=int(os.getenv("POSTGRES_PORT", "6432")),
            database=os.getenv("POSTGRES_DB", "startml"),
            user=os.getenv("POSTGRES_USER", "robot-startml-ro"),
            password=os.getenv("POSTGRES_PASSWORD"),
        )
    except Exception as e:
        print("Ошибка при подключении к базе данных.")
        raise e

    conn.autocommit = True

    return conn
