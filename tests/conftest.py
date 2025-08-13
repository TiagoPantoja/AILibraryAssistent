import pytest
import json
import tempfile
import os
from typing import List, Dict, Any
from pathlib import Path

from models.books import Book, BookResponse
from models.chat import ChatRequest, ChatResponse
from services.book_service import BookService
from services.nlp_pipeline import NLPPipeline
from services.recommendation_engine import RecommendationEngine


@pytest.fixture
def sample_book_data() -> Dict[str, Any]:
    """Dados de exemplo para um livro"""
    return {
        "id": 1,
        "title": "Test Book",
        "author": "Test Author",
        "genre": "Test Genre",
        "year": 2023,
        "bestseller": True,
        "description": "A test book description",
        "rating": 4.5
    }


@pytest.fixture
def sample_book(sample_book_data) -> Book:
    """Instância de Book para testes"""
    return Book(**sample_book_data)


@pytest.fixture
def sample_books_list() -> List[Dict[str, Any]]:
    """Lista de livros para testes"""
    return [
        {
            "id": 1,
            "title": "Terror Book",
            "author": "Horror Author",
            "genre": "Terror",
            "year": 2020,
            "bestseller": True,
            "description": "A scary book",
            "rating": 4.2
        },
        {
            "id": 2,
            "title": "Romance Book",
            "author": "Love Author",
            "genre": "Romance",
            "year": 2021,
            "bestseller": False,
            "description": "A romantic book",
            "rating": 3.8
        },
        {
            "id": 3,
            "title": "Sci-Fi Book",
            "author": "Future Author",
            "genre": "Ficção Científica",
            "year": 2022,
            "bestseller": True,
            "description": "A futuristic book",
            "rating": 4.6
        }
    ]


@pytest.fixture
def sample_books_objects(sample_books_list) -> List[Book]:
    """Lista de objetos Book para testes"""
    return [Book(**book_data) for book_data in sample_books_list]


@pytest.fixture
def sample_chat_request() -> ChatRequest:
    """Request de chat para testes"""
    return ChatRequest(
        message="Gosto de livros de terror",
        user_id="test_user_123"
    )


@pytest.fixture
def sample_chat_response() -> ChatResponse:
    """Response de chat para testes"""
    return ChatResponse(
        response="Aqui estão algumas recomendações de terror:",
        recommended_books=[],
        intent="recommend_by_genre",
        confidence=0.85
    )


@pytest.fixture
def temp_books_json(sample_books_list):
    """Arquivo JSON temporário para testes"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"books": sample_books_list}, f)
        temp_path = f.name

    yield temp_path

    # Cleanup
    os.unlink(temp_path)


@pytest.fixture
def book_service(temp_books_json):
    """BookService configurado para testes"""
    return BookService(data_path=temp_books_json)


@pytest.fixture
def nlp_pipeline():
    """NLPPipeline para testes"""
    return NLPPipeline()


@pytest.fixture
def recommendation_engine(book_service):
    """RecommendationEngine para testes"""
    return RecommendationEngine(book_service)