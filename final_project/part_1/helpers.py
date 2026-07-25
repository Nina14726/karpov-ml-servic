from typing import List, Optional
from psycopg2.extensions import connection
from psycopg2.extras import DictCursor

from models import User, Post, Feed


def get_user(conn: connection, user_id: int) -> Optional[User]:
    query = """
        SELECT id, gender, age, country, city, exp_group, os, source
        FROM public.user
        WHERE id = %s
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (user_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return User(**row)


def get_post(conn: connection, post_id: int) -> Optional[Post]:
    query = """
        SELECT id, text, topic
        FROM public.post
        WHERE id = %s
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (post_id,))
        row = cur.fetchone()
        if row is None:
            return None
        return Post(**row)


def get_feed(
    conn: connection, user_id: int = None, post_id: int = None, limit: int = 10
) -> List[Feed]:
    if user_id is None and post_id is None:
        raise ValueError("Необходимо указать хотя бы user_id или post_id")

    query = """
        SELECT
            fa.user_id, fa.post_id, fa.action, fa.time,
            u.id, u.gender, u.age, u.country, u.city, u.exp_group, u.os, u.source,
            p.id, p.text, p.topic
        FROM public.feed_action fa
        JOIN public.user u ON fa.user_id = u.id
        JOIN public.post p ON fa.post_id = p.id
        WHERE 1=1
    """

    params = []
    if user_id is not None:
        query += " AND fa.user_id = %s"
        params.append(user_id)
    if post_id is not None:
        query += " AND fa.post_id = %s"
        params.append(post_id)

    query += " ORDER BY fa.time DESC LIMIT %s"
    params.append(limit)

    result = []
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, params)
        rows = cur.fetchall()
        for row in rows:
            user = User(
                id=row["user_id"],
                gender=row["gender"],
                age=row["age"],
                country=row["country"],
                city=row["city"],
                exp_group=row["exp_group"],
                os=row["os"],
                source=row["source"],
            )
            post = Post(
                id=row["post_id"],
                text=row["text"],
                topic=row["topic"],
            )
            feed = Feed(
                user_id=row["user_id"],
                post_id=row["post_id"],
                user=user,
                post=post,
                action=row["action"],
                time=row["time"],
            )
            result.append(feed)

    return result


def get_recommended_feed(conn: connection, id: int, limit: int) -> List[Post]:
    query = """
        SELECT p.id, p.text, p.topic
        FROM public.post p
        JOIN public.feed_action fa ON p.id = fa.post_id
        WHERE fa.action = 'like'
        GROUP BY p.id, p.text, p.topic
        ORDER BY COUNT(*) DESC
        LIMIT %s
    """
    with conn.cursor(cursor_factory=DictCursor) as cur:
        cur.execute(query, (limit,))
        rows = cur.fetchall()
        return [Post(**row) for row in rows]
