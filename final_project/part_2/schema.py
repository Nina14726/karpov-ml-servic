from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class UserGet(BaseModel):
    """Модель данных пользователя для API-ответов."""

    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str


class PostGet(BaseModel):
    """Модель данных поста для API-ответов."""

    id: int
    text: str
    topic: Optional[str] = None


class FeedGet(BaseModel):
    """Модель действия пользователя с постом для API-ответов."""

    user_id: int
    post_id: int
    user: UserGet
    post: PostGet
    action: str
    time: datetime
