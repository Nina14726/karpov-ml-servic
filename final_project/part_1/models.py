from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class User:
    id: int
    gender: int
    age: int
    country: str
    city: str
    exp_group: int
    os: str
    source: str


@dataclass
class Post:
    id: int
    text: str
    topic: Optional[str] = None


@dataclass
class Feed:
    user_id: int
    post_id: int
    user: User
    post: Post
    action: str
    time: datetime
