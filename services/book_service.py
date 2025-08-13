import json
from typing import List, Optional
from models.books import Book


class BookService:
    def __init__(self, data_path: str = "data/books.json"):
        self.data_path = data_path
        self.books = self._load_books()

    def _load_books(self) -> List[Book]:
        """Carrega os livros do arquivo JSON"""
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                return [Book(**book) for book in data['books']]
        except FileNotFoundError:
            return []

    def get_all_books(self) -> List[Book]:
        """Retorna todos os livros"""
        return self.books

    def get_book_by_id(self, book_id: int) -> Optional[Book]:
        """Retorna um livro específico por ID"""
        for book in self.books:
            if book.id == book_id:
                return book
        return None

    def get_books_by_genre(self, genre: str) -> List[Book]:
        """Retorna livros por gênero"""
        return [book for book in self.books
                if book.genre.lower() == genre.lower()]

    def get_books_by_author(self, author: str) -> List[Book]:
        """Retorna livros por autor"""
        return [book for book in self.books
                if author.lower() in book.author.lower()]

    def get_books_by_year(self, year: int) -> List[Book]:
        """Retorna livros por ano"""
        return [book for book in self.books if book.year == year]

    def get_bestsellers(self, is_bestseller: bool = True) -> List[Book]:
        """Retorna bestsellers ou não-bestsellers"""
        return [book for book in self.books
                if book.bestseller == is_bestseller]

    def find_book_by_title(self, title: str) -> Optional[Book]:
        """Encontra um livro pelo título"""
        for book in self.books:
            if title.lower() in book.title.lower():
                return book
        return None