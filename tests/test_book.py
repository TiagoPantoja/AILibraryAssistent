import pytest
from pydantic import ValidationError
from typing import List

from models.books import Book, BookResponse

class TestBook:
    """Testes para o modelo Book"""


def test_book_creation_valid_data(self, sample_book_data):
    """Teste criação de Book com dados válidos"""
    book = Book(**sample_book_data)

    assert book.id == sample_book_data["id"]
    assert book.title == sample_book_data["title"]
    assert book.author == sample_book_data["author"]
    assert book.genre == sample_book_data["genre"]
    assert book.year == sample_book_data["year"]
    assert book.bestseller == sample_book_data["bestseller"]
    assert book.description == sample_book_data["description"]
    assert book.rating == sample_book_data["rating"]


def test_book_creation_from_existing_book(self, sample_book):
    """Teste criação de Book a partir de outro Book"""
    new_book = Book(**sample_book.model_dump())

    assert new_book.id == sample_book.id
    assert new_book.title == sample_book.title
    assert new_book.author == sample_book.author
    assert new_book == sample_book


@pytest.mark.parametrize("field,invalid_value,expected_error", [
    ("id", "not_an_int", "Input should be a valid integer"),
    ("id", -1, None),
    ("title", "", None),
    ("title", 123, "Input should be a valid string"),
    ("author", None, "Input should be a valid string"),
    ("genre", [], "Input should be a valid string"),
    ("year", "2023", None),
    ("year", "abc", "Input should be a valid integer"),
    ("bestseller", "true", None),
    ("bestseller", "invalid", "Input should be a valid boolean"),
    ("description", 123, "Input should be a valid string"),
    ("rating", "4.5", None),
    ("rating", "invalid", "Input should be a valid number"),
    ("rating", -1.0, None),
    ("rating", 6.0, None),
])
def test_book_validation_errors(self, sample_book_data, field, invalid_value, expected_error):
    """Teste validação de campos inválidos"""
    invalid_data = sample_book_data.copy()
    invalid_data[field] = invalid_value

    if expected_error:
        with pytest.raises(ValidationError) as exc_info:
            Book(**invalid_data)
        assert expected_error in str(exc_info.value)
    else:
        # Deve criar sem erro
        book = Book(**invalid_data)
        assert getattr(book, field) is not None


def test_book_missing_required_fields(self):
    """Teste criação de Book com campos obrigatórios faltando"""
    required_fields = ["id", "title", "author", "genre", "year", "bestseller", "description", "rating"]

    for field in required_fields:
        incomplete_data = {
            "id": 1,
            "title": "Test",
            "author": "Author",
            "genre": "Genre",
            "year": 2023,
            "bestseller": True,
            "description": "Description",
            "rating": 4.0
        }
        del incomplete_data[field]

        with pytest.raises(ValidationError) as exc_info:
            Book(**incomplete_data)
        assert "Field required" in str(exc_info.value)


def test_book_extra_fields_ignored(self, sample_book_data):
    """Teste que campos extras são ignorados"""
    data_with_extra = sample_book_data.copy()
    data_with_extra["extra_field"] = "should_be_ignored"
    data_with_extra["another_extra"] = 123

    book = Book(**data_with_extra)

    assert not hasattr(book, "extra_field")
    assert not hasattr(book, "another_extra")
    assert book.title == sample_book_data["title"]


def test_book_model_dump(self, sample_book):
    """Teste serialização do modelo Book"""
    book_dict = sample_book.model_dump()

    assert isinstance(book_dict, dict)
    assert book_dict["id"] == sample_book.id
    assert book_dict["title"] == sample_book.title
    assert book_dict["author"] == sample_book.author
    assert book_dict["genre"] == sample_book.genre
    assert book_dict["year"] == sample_book.year
    assert book_dict["bestseller"] == sample_book.bestseller
    assert book_dict["description"] == sample_book.description
    assert book_dict["rating"] == sample_book.rating


def test_book_model_dump_json(self, sample_book):
    """Teste serialização JSON do modelo Book"""
    book_json = sample_book.model_dump_json()

    assert isinstance(book_json, str)
    assert '"id":' in book_json
    assert '"title":' in book_json
    assert sample_book.title in book_json


def test_book_equality(self, sample_book_data):
    """Teste igualdade entre objetos Book"""
    book1 = Book(**sample_book_data)
    book2 = Book(**sample_book_data)

    assert book1 == book2
    assert book1 is not book2

    modified_data = sample_book_data.copy()
    modified_data["title"] = "Different Title"
    book3 = Book(**modified_data)

    assert book1 != book3


def test_book_hash(self, sample_book_data):
    """Teste hash de objetos Book"""
    book1 = Book(**sample_book_data)
    book2 = Book(**sample_book_data)

    assert hash(book1) == hash(book2)

    book_set = {book1, book2}
    assert len(book_set) == 1


def test_book_string_representation(self, sample_book):
    """Teste representação string do Book"""
    book_str = str(sample_book)
    book_repr = repr(sample_book)

    assert sample_book.title in book_str or sample_book.title in book_repr
    assert "Book" in book_repr


@pytest.mark.parametrize("rating,expected_valid", [
    (0.0, True),
    (2.5, True),
    (5.0, True),
    (5.1, True),
    (-0.1, True),
])
def test_book_rating_edge_cases(self, sample_book_data, rating, expected_valid):
    """Teste casos extremos de rating"""
    data = sample_book_data.copy()
    data["rating"] = rating

    if expected_valid:
        book = Book(**data)
        assert book.rating == rating
    else:
        with pytest.raises(ValidationError):
            Book(**data)


@pytest.mark.parametrize("year,expected_valid", [
    (1, True),
    (1000, True),
    (2023, True),
    (2050, True),
    (0, True),
    (-100, True),
])
def test_book_year_edge_cases(self, sample_book_data, year, expected_valid):
    """Teste casos extremos de ano"""
    data = sample_book_data.copy()
    data["year"] = year

    if expected_valid:
        book = Book(**data)
        assert book.year == year
    else:
        with pytest.raises(ValidationError):
            Book(**data)
