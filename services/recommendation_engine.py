from typing import List, Dict, Optional
from models.books import Book
from services.book_service import BookService
import math


class RecommendationEngine:
    """Engine de recomendação de livros baseado em diferentes critérios."""

    def __init__(self, book_service: BookService):
        """
        Inicializa o engine de recomendação.

        Args:
            book_service: Serviço para operações com livros
        """
        self.book_service = book_service

    def recommend_by_genre(self, genre: str, limit: int = 5) -> List[Book]:
        """
        Recomenda livros por gênero.

        Args:
            genre: Gênero dos livros a serem recomendados
            limit: Número máximo de livros a retornar

        Returns:
            Lista de livros ordenados por rating
        """
        books = self.book_service.get_books_by_genre(genre)
        return sorted(books, key=lambda x: x.rating, reverse=True)[:limit]

    def recommend_by_author(self, author: str, limit: int = 5) -> List[Book]:
        """
        Recomenda livros por autor.

        Args:
            author: Nome do autor
            limit: Número máximo de livros a retornar

        Returns:
            Lista de livros do autor ordenados por rating
        """
        books = self.book_service.get_books_by_author(author)
        return sorted(books, key=lambda x: x.rating, reverse=True)[:limit]

    def recommend_similar_books(self, book_title: str, limit: int = 5) -> List[Book]:
        """
        Recomenda livros similares baseado no gênero e autor.

        Args:
            book_title: Título do livro de referência
            limit: Número máximo de livros a retornar

        Returns:
            Lista de livros similares ordenados por rating
        """
        target_book = self.book_service.find_book_by_title(book_title)

        if not target_book:
            return []

        # Busca livros do mesmo gênero
        similar_books = self.book_service.get_books_by_genre(target_book.genre)

        # Remove o próprio livro da lista
        similar_books = [
            book for book in similar_books
            if book.id != target_book.id
        ]

        # Prioriza livros do mesmo autor
        same_author_books = [
            book for book in similar_books
            if book.author == target_book.author
        ]
        other_books = [
            book for book in similar_books
            if book.author != target_book.author
        ]

        # Combina e ordena por rating
        recommended = same_author_books + other_books
        return sorted(recommended, key=lambda x: x.rating, reverse=True)[:limit]

    def recommend_by_year(self, year: int, limit: int = 5) -> List[Book]:
        """
        Recomenda livros por ano de publicação.

        Args:
            year: Ano de publicação
            limit: Número máximo de livros a retornar

        Returns:
            Lista de livros do ano ordenados por rating
        """
        books = self.book_service.get_books_by_year(year)
        return sorted(books, key=lambda x: x.rating, reverse=True)[:limit]

    def recommend_bestsellers(self, is_bestseller: bool = True, limit: int = 5) -> List[Book]:
        """
        Recomenda bestsellers ou não-bestsellers.

        Args:
            is_bestseller: Se True, retorna bestsellers; se False, não-bestsellers
            limit: Número máximo de livros a retornar

        Returns:
            Lista de livros ordenados por rating
        """
        books = self.book_service.get_bestsellers(is_bestseller)
        return sorted(books, key=lambda x: x.rating, reverse=True)[:limit]

    def recommend_by_mood(self, mood: str, target_genre: str = None, limit: int = 5) -> List[Book]:
        """
        Recomenda livros baseado no humor/estado emocional do usuário.

        Args:
            mood: Estado emocional do usuário
            target_genre: Gênero específico (opcional)
            limit: Número máximo de livros a retornar

        Returns:
            Lista de livros apropriados para o humor
        """
        # Se não foi passado um gênero alvo, usa o gênero padrão baseado no humor
        if not target_genre:
            mood_to_genre_map = {
                'triste': 'Romance',
                'deprimido': 'Romance',
                'estressado': 'Romance',
                'ansioso': 'Romance',
                'feliz': 'Thriller',
                'animado': 'Thriller',
                'relaxar': 'Romance',
                'inspirar': 'Ficção Científica'
            }
            target_genre = mood_to_genre_map.get(mood.lower(), 'Romance')

        # Busca livros do gênero apropriado
        books = self.book_service.get_books_by_genre(target_genre)

        # Para humores "negativos", prioriza livros com ratings mais altos
        negative_moods = ['triste', 'deprimido', 'estressado', 'ansioso', 'mal']
        if mood.lower() in negative_moods:
            # Filtra apenas livros bem avaliados para garantir uma boa experiência
            books = [book for book in books if book.rating >= 4.0]

        return sorted(books, key=lambda x: x.rating, reverse=True)[:limit]

    def recommend_by_occasion(self, occasion: str, target_genre: str = None, limit: int = 5) -> List[Book]:
        """
        Recomenda livros baseado na ocasião.

        Args:
            occasion: Ocasião ou contexto de leitura
            target_genre: Gênero específico (opcional)
            limit: Número máximo de livros a retornar

        Returns:
            Lista de livros apropriados para a ocasião
        """
        if not target_genre:
            target_genre = 'Romance'  # Padrão para ocasiões casuais

        books = self.book_service.get_books_by_genre(target_genre)

        # Para ocasiões de relaxamento, prioriza livros mais leves
        relaxing_occasions = ['praia', 'férias', 'viagem', 'relaxar']
        if occasion.lower() in relaxing_occasions:
            # Prioriza livros com boa avaliação mas não muito "pesados"
            books = [book for book in books if book.rating >= 3.5]

        return sorted(books, key=lambda x: x.rating, reverse=True)[:limit]

    def calculate_similarity(self, book1: Book, book2: Book) -> float:
        """
        Calcula similaridade entre dois livros.

        Args:
            book1: Primeiro livro para comparação
            book2: Segundo livro para comparação

        Returns:
            Valor de similaridade entre 0 e 1
        """
        similarity = 0.0

        # Mesmo gênero (peso: 40%)
        if book1.genre == book2.genre:
            similarity += 0.4

        # Mesmo autor (peso: 30%)
        if book1.author == book2.author:
            similarity += 0.3

        # Similaridade de rating (peso: 20%)
        rating_diff = abs(book1.rating - book2.rating)
        rating_similarity = max(0, 1 - (rating_diff / 5))
        similarity += rating_similarity * 0.2

        # Similaridade de ano (peso: 10%)
        year_diff = abs(book1.year - book2.year)
        year_similarity = max(0, 1 - (year_diff / 50))
        similarity += year_similarity * 0.1

        return similarity