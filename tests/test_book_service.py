import pytest
import json
import tempfile
import os
from unittest.mock import patch, mock_open, MagicMock
from typing import List

from models.books import Book
from services.book_service import BookService

class TestBookService:
    """Testes para a classe BookService"""


def test_book_service_initialization_with_valid_file(self, temp_books_json):
    """Teste inicialização do BookService com arquivo válido"""
    service = BookService(data_path=temp_books_json)

    assert service.data_path == temp_books_json
    assert isinstance(service.books, list)
    assert len(service.books) > 0
    assert all(isinstance(book, Book) for book in service.books)


def test_book_service_initialization_with_nonexistent_file(self):
    """Teste inicialização do BookService com arquivo inexistente"""
    service = BookService(data_path="nonexistent_file.json")

    assert service.data_path == "nonexistent_file.json"
    assert service.books == []


def test_book_service_initialization_default_path(self):
    """Teste inicialização do BookService com caminho padrão"""
    with patch('builtins.open', mock_open(read_data='{"books": []}')):
        service = BookService()

        assert service.data_path == "data/books.json"
        assert service.books == []


@patch('builtins.open', side_effect=FileNotFoundError)
def test_load_books_file_not_found(self, mock_file):
    """Teste _load_books quando arquivo não existe"""
    service = BookService(data_path="missing_file.json")

    assert service.books == []
    mock_file.assert_called_once()


@patch('builtins.open', mock_open(read_data='invalid json'))
def test_load_books_invalid_json(self):
    """Teste _load_books com JSON inválido"""
    with pytest.raises(json.JSONDecodeError):
        BookService(data_path="invalid.json")


@patch('builtins.open', mock_open(read_data='{"wrong_key": []}'))
def test_load_books_missing_books_key(self):
    """Teste _load_books com estrutura JSON incorreta"""
    with pytest.raises(KeyError):
        BookService(data_path="wrong_structure.json")


def test_load_books_with_invalid_book_data(self):
    """Teste _load_books com dados de livro inválidos"""
    invalid_data = {
        "books": [
            {
                "id": 1,
                "title": "Valid Book",
                "author": "Valid Author",
                "genre": "Valid Genre",
                "year": 2023,
                "bestseller": True,
                "description": "Valid description",
                "rating": 4.5
            },
            {
                "id": "invalid_id",  # ID inválido
                "title": "Invalid Book"
                # Campos obrigatórios faltando
            }
        ]
    }

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(invalid_data, f)
        temp_path = f.name

    try:
        with pytest.raises(Exception):  # Pode ser ValidationError ou TypeError
            BookService(data_path=temp_path)
    finally:
        os.unlink(temp_path)

    def test_get_all_books(self, book_service, sample_books_objects):
        """Teste get_all_books retorna todos os livros"""
        books = book_service.get_all_books()

        assert isinstance(books, list)
        assert len(books) == len(sample_books_objects)
        assert all(isinstance(book, Book) for book in books)

    def test_get_all_books_empty_service(self):
        """Teste get_all_books com serviço vazio"""
        with patch('builtins.open', mock_open(read_data='{"books": []}')):
            service = BookService()
            books = service.get_all_books()

            assert books == []

    def test_get_all_books_returns_copy(self, book_service):
        """Teste que get_all_books retorna a lista original (não uma cópia)"""
        books1 = book_service.get_all_books()
        books2 = book_service.get_all_books()

        # Deve retornar a mesma lista (referência)
        assert books1 is books2

    @pytest.mark.parametrize("book_id,expected_found", [
        (1, True),
        (2, True),
        (3, True),
        (999, False),
        (0, False),
        (-1, False),
    ])
    def test_get_book_by_id(self, book_service, book_id, expected_found):
        """Teste get_book_by_id com vários IDs"""
        book = book_service.get_book_by_id(book_id)

        if expected_found:
            assert book is not None
            assert isinstance(book, Book)
            assert book.id == book_id
        else:
            assert book is None

    def test_get_book_by_id_type_validation(self, book_service):
        """Teste get_book_by_id com tipos inválidos"""
        # O método deve funcionar com tipos que podem ser comparados
        assert book_service.get_book_by_id("1") is None  # String não encontra int
        assert book_service.get_book_by_id(1.0) is None  # Float não encontra int exato

    @pytest.mark.parametrize("genre,expected_count", [
        ("Terror", 1),
        ("terror", 1),
        ("TERROR", 1),
        ("Romance", 1),
        ("Ficção Científica", 1),
        ("Inexistente", 0),
        ("", 0),
    ])
    def test_get_books_by_genre(self, book_service, genre, expected_count):
        """Teste get_books_by_genre com vários gêneros"""
        books = book_service.get_books_by_genre(genre)

        assert isinstance(books, list)
        assert len(books) == expected_count

        if expected_count > 0:
            assert all(book.genre.lower() == genre.lower() for book in books)

    def test_get_books_by_genre_case_insensitive(self, book_service):
        """Teste que get_books_by_genre é case insensitive"""
        books_lower = book_service.get_books_by_genre("terror")
        books_upper = book_service.get_books_by_genre("TERROR")
        books_mixed = book_service.get_books_by_genre("Terror")

        assert len(books_lower) == len(books_upper) == len(books_mixed)
        if books_lower:
            assert books_lower[0].id == books_upper[0].id == books_mixed[0].id

    @pytest.mark.parametrize("author,expected_count", [
        ("Horror Author", 1),
        ("horror author", 1),
        ("HORROR AUTHOR", 1),
        ("Horror", 1),
        ("Author", 3),
        ("Love", 1),
        ("Inexistente", 0),
        ("", 3),
    ])
    def test_get_books_by_author(self, book_service, author, expected_count):
        """Teste get_books_by_author com vários autores"""
        books = book_service.get_books_by_author(author)

        assert isinstance(books, list)
        assert len(books) == expected_count

        if expected_count > 0:
            assert all(author.lower() in book.author.lower() for book in books)

    def test_get_books_by_author_partial_match(self, book_service):
        """Teste que get_books_by_author faz busca parcial"""
        # Busca por parte do nome
        books = book_service.get_books_by_author("Love")

        assert len(books) == 1
        assert "Love Author" in books[0].author

    @pytest.mark.parametrize("year,expected_count", [
        (2020, 1),
        (2021, 1),
        (2022, 1),
        (1999, 0),
        (2025, 0),
        (0, 0),
    ])
    def test_get_books_by_year(self, book_service, year, expected_count):
        """Teste get_books_by_year com vários anos"""
        books = book_service.get_books_by_year(year)

        assert isinstance(books, list)
        assert len(books) == expected_count

        if expected_count > 0:
            assert all(book.year == year for book in books)

    def test_get_books_by_year_type_validation(self, book_service):
        """Teste get_books_by_year com tipos diferentes"""
        # Deve funcionar apenas com int exato
        books_int = book_service.get_books_by_year(2020)
        books_float = book_service.get_books_by_year(2020.0)

        assert len(books_int) == 1
        assert len(books_float) == 0  # Float não é igual a int

    @pytest.mark.parametrize("is_bestseller,expected_count", [
        (True, 2),  # Terror e Sci-Fi são bestsellers
        (False, 1),  # Romance não é bestseller
    ])
    def test_get_bestsellers(self, book_service, is_bestseller, expected_count):
        """Teste get_bestsellers com True e False"""
        books = book_service.get_bestsellers(is_bestseller)

        assert isinstance(books, list)
        assert len(books) == expected_count
        assert all(book.bestseller == is_bestseller for book in books)

    def test_get_bestsellers_default_parameter(self, book_service):
        """Teste get_bestsellers com parâmetro padrão"""
        books_default = book_service.get_bestsellers()
        books_explicit = book_service.get_bestsellers(True)

        assert len(books_default) == len(books_explicit)
        assert books_default == books_explicit

    @pytest.mark.parametrize("title,expected_found", [
        ("Terror Book", True),
        ("terror book", True),  # Case insensitive
        ("TERROR BOOK", True),  # Case insensitive
        ("Terror", True),  # Partial match
        ("Book", True),  # Partial match - primeiro encontrado
        ("Romance", True),  # Partial match
        ("Inexistente", False),
        ("", True),  # String vazia está em todas as strings
    ])
    def test_find_book_by_title(self, book_service, title, expected_found):
        """Teste find_book_by_title com vários títulos"""
        book = book_service.find_book_by_title(title)

        if expected_found:
            assert book is not None
            assert isinstance(book, Book)
            assert title.lower() in book.title.lower()
        else:
            assert book is None

    def test_find_book_by_title_returns_first_match(self, book_service):
        """Teste que find_book_by_title retorna o primeiro match"""
        # "Book" aparece em todos os títulos
        book = book_service.find_book_by_title("Book")

        assert book is not None
        # Deve retornar o primeiro livro (ID 1)
        assert book.id == 1

    def test_find_book_by_title_case_insensitive(self, book_service):
        """Teste que find_book_by_title é case insensitive"""
        book_lower = book_service.find_book_by_title("terror book")
        book_upper = book_service.find_book_by_title("TERROR BOOK")
        book_mixed = book_service.find_book_by_title("Terror Book")

        assert book_lower is not None
        assert book_upper is not None
        assert book_mixed is not None
        assert book_lower.id == book_upper.id == book_mixed.id