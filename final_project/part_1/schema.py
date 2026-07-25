from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class UserGet(BaseModel):
    """
    Модель данных пользователя для API-ответов.

    Атрибуты:
    - id: уникальный идентификатор пользователя.
    - gender: пол пользователя, представлен числом
    - age: возраст пользователя.
    - country: страна проживания.
    - city: город проживания.
    - exp_group: группа эксперимента, к которой относится пользователь.
    - os: операционная система устройства пользователя
    - source: источник регистрации или привлечения пользователя.
    """
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str


class PostGet(BaseModel):
    """
    Модель данных поста для API-ответов.

    Атрибуты:
    - id: уникальный идентификатор поста.
    - text: текстовое содержимое поста.
    - topic: тема или категория поста.
    """
    id: int
    text: str
    topic: Optional[str] = None


class FeedGet(BaseModel):
    """
    Модель данных действия пользователя с постом для API-ответов.

    Атрибуты:
    - user_id: идентификатор пользователя, который совершил действие.
    - post_id: идентификатор поста, над которым совершено действие.
    - user: вложенный объект UserGet с данными пользователя.
    - post: вложенный объект PostGet с данными поста.
    - action: тип действия
    - time: дата и время совершения действия в формате datetime.
    """
    user_id: int
    post_id: int
    user: UserGet
    post: PostGet
    action: str
    time: datetime
