import pytest
from unittest.mock import Mock, MagicMock, patch
from typing import List

from models.books import Book
from services.book_service import BookService
from services.recommendation_engine import RecommendationEngine

class TestRecommendationEngineInitialization:
    """Testes para inicialização do RecommendationEngine"""

    def test_recommendation_engine_initialization(self):
        """Teste inicialização básica do RecommendationEngine"""
        mock_book_service = Mock(spec=BookService)
        engine = RecommendationEngine(mock_book_service)

        assert engine.book_service is mock_book_service

    def test_recommendation_engine_with_real_book_service(self, book_service):
        """Teste inicialização com BookService real"""
        engine = RecommendationEngine(book_service)

        assert engine.book_service is book_service
        assert hasattr(engine, 'book_service')

    @pytest.fixture
    def mock_books(self):
        """Livros mock para testes"""
        return [
            Book(id=1, title="Terror Book 1", author="Author 1", genre="Terror",
                 year=2020, bestseller=True, description="Desc 1", rating=4.5),
            Book(id=2, title="Terror Book 2", author="Author 2", genre="Terror",
                 year=2021, bestseller=False, description="Desc 2", rating=4.2),
            Book(id=3, title="Terror Book 3", author="Author 3", genre="Terror",
                 year=2022, bestseller=True, description="Desc 3", rating=4.8),
        ]

    def test_recommend_by_genre_success(self, mock_books):
        """Teste recomendação por gênero com sucesso"""
        mock_book_service = Mock(spec=BookService)
        mock_book_service.get_books_by_genre.return_value = mock_books

        engine = RecommendationEngine(mock_book_service)
        result = engine.recommend_by_genre("Terror")

        mock_book_service.get_books_by_genre.assert_called_once_with("Terror")
        assert len(result) == 3
        # Deve estar ordenado por rating (decrescente)
        assert result[0].rating == 4.8
        assert result[1].rating == 4.5
        assert result[2].rating == 4.2

    def test_recommend_by_genre_with_limit(self, mock_books):
        """Teste recomendação por gênero com limite"""
        mock_book_service = Mock(spec=BookService)
        mock_book_service.get_books_by_genre.return_value = mock_books

        engine = RecommendationEngine(mock_book_service)
        result = engine.recommend_by_genre("Terror", limit=2)

        assert len(result) == 2
        assert result[0].rating == 4.8
        assert result[1].rating == 4.5

    def test_recommend_by_genre_empty_result(self):
        """Teste recomendação por gênero sem resultados"""
        mock_book_service = Mock(spec=BookService)
        mock_book_service.get_books_by_genre.return_value = []

        engine = RecommendationEngine(mock_book_service)
        result = engine.recommend_by_genre("Inexistente")

        assert result == []

    def test_recommend_by_genre_default_limit(self, mock_books):
        """Teste limite padrão de recomendação por gênero"""
        # Cria mais de 5 livros para testar o limite padrão
        many_books = []
        for i in range(10):
            many_books.append(
                Book(id=i, title=f"Book {i}", author=f"Author {i}", genre="Terror",
                     year=2020, bestseller=True, description=f"Desc {i}", rating=4.0 + i * 0.1)
            )

        mock_book_service = Mock(spec=BookService)
        mock_book_service.get_books_by_genre.return_value = many_books

        engine = RecommendationEngine(mock_book_service)
        result = engine.recommend_by_genre("Terror")

        assert len(result) == 5  # Limite padrão

    @pytest.mark.parametrize("genre", ["Terror", "Romance", "Ficção Científica", ""])
    def test_recommend_by_genre_different_genres(self, genre):
        """Teste recomendação com diferentes gêneros"""
        mock_book_service = Mock(spec=BookService)
        mock_book_service.get_books_by_genre.return_value = []

        engine = RecommendationEngine(mock_book_service)
        result = engine.recommend_by_genre(genre)

        mock_book_service.get_books_by_genre.assert_called_once_with(genre)
        assert result == []