from pydantic import BaseModel
from typing import Optional

class Book(BaseModel):
    id: int
    title: str
    author: str
    genre: str
    year: int
    bestseller: bool
    description: str
    rating: float

class BookResponse(BaseModel):
    books: list[Book]
    total: int