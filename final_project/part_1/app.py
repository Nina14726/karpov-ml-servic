from typing import List
from fastapi import FastAPI, HTTPException, Depends
from database import postgres_connection
from schema import UserGet, PostGet, FeedGet
from helpers import get_user, get_post, get_feed, get_recommended_feed

app = FastAPI()


def get_conn():
    conn = postgres_connection()
    try:
        yield conn
    finally:
        conn.close()


@app.get("/user/{id}", response_model=UserGet)
def handle_get_user(id: int, conn=Depends(get_conn)) -> UserGet:
    user = get_user(conn, id)
    if user is None:
        raise HTTPException(status_code=404, detail="user not found")
    return user


@app.get("/post/{id}", response_model=PostGet)
def handle_get_post(id: int, conn=Depends(get_conn)) -> PostGet:
    post = get_post(conn, id)
    if post is None:
        raise HTTPException(status_code=404, detail="post not found")
    return post


@app.get("/user/{id}/feed", response_model=List[FeedGet])
def handle_get_user_feed(id: int, limit: int = 10, conn=Depends(get_conn)) -> List[FeedGet]:
    return get_feed(conn, user_id=id, limit=limit)


@app.get("/post/{id}/feed", response_model=List[FeedGet])
def handle_get_post_feed(id: int, limit: int = 10, conn=Depends(get_conn)) -> List[FeedGet]:
    return get_feed(conn, post_id=id, limit=limit)


@app.get("/post/recommendations/", response_model=List[PostGet])
def recommended_posts(id: int, limit: int = 10, conn=Depends(get_conn)) -> List[PostGet]:
    return get_recommended_feed(conn, id, limit)
