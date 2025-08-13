import pytest
import re
from unittest.mock import patch, MagicMock
from typing import Dict, List

from services.nlp_pipeline import NLPPipeline, Intent

class TestIntent:
    """Testes para a classe Intent (dataclass)"""

    def test_intent_creation(self):
        """Teste criação básica de Intent"""
        intent = Intent(
            name="recommend_by_genre",
            confidence=0.85,
            entities={"target": "terror"}
        )

        assert intent.name == "recommend_by_genre"
        assert intent.confidence == 0.85
        assert intent.entities == {"target": "terror"}

    def test_intent_equality(self):
        """Teste igualdade entre objetos Intent"""
        intent1 = Intent("test", 0.5, {"key": "value"})
        intent2 = Intent("test", 0.5, {"key": "value"})
        intent3 = Intent("different", 0.5, {"key": "value"})

        assert intent1 == intent2
        assert intent1 != intent3

    def test_intent_string_representation(self):
        """Teste representação string de Intent"""
        intent = Intent("test_intent", 0.75, {"entity": "value"})

        intent_str = str(intent)
        intent_repr = repr(intent)

        assert "test_intent" in intent_str or "test_intent" in intent_repr
        assert "0.75" in intent_str or "0.75" in intent_repr

        def test_nlp_pipeline_initialization(self):
            """Teste inicialização básica do NLPPipeline"""
            pipeline = NLPPipeline()

            assert hasattr(pipeline, 'intent_patterns')
            assert hasattr(pipeline, 'mood_to_genre')
            assert hasattr(pipeline, 'occasion_to_genre')
            assert hasattr(pipeline, 'genre_synonyms')
            assert hasattr(pipeline, 'famous_authors')
            assert hasattr(pipeline, 'confidence_threshold')
            assert hasattr(pipeline, 'advanced_processing_enabled')

        def test_intent_patterns_structure(self):
            """Teste estrutura dos padrões de intenção"""
            pipeline = NLPPipeline()

            expected_intents = [
                'recommend_by_genre',
                'recommend_by_author',
                'recommend_similar',
                'books_by_year',
                'bestsellers',
                'non_bestsellers',
                'recommend_by_mood',
                'recommend_by_occasion'
            ]

            for intent in expected_intents:
                assert intent in pipeline.intent_patterns
                assert isinstance(pipeline.intent_patterns[intent], list)
                assert len(pipeline.intent_patterns[intent]) > 0

        def test_mood_to_genre_mapping(self):
            """Teste mapeamento de humor para gênero"""
            pipeline = NLPPipeline()

            # Testa alguns mapeamentos específicos
            assert pipeline.mood_to_genre['triste'] == 'Romance'
            assert pipeline.mood_to_genre['feliz'] == 'Fantasia'
            assert pipeline.mood_to_genre['reflexivo'] == 'Filosofia'
            assert pipeline.mood_to_genre['emocionante'] == 'Thriller'

        def test_occasion_to_genre_mapping(self):
            """Teste mapeamento de ocasião para gênero"""
            pipeline = NLPPipeline()

            # Testa alguns mapeamentos específicos
            assert pipeline.occasion_to_genre['viajar'] == 'Romance'
            assert pipeline.occasion_to_genre['trabalhar'] == 'Autoajuda'
            assert pipeline.occasion_to_genre['estudar'] == 'História'
            assert pipeline.occasion_to_genre['avião'] == 'Mistério'

        def test_genre_synonyms_structure(self):
            """Teste estrutura dos sinônimos de gênero"""
            pipeline = NLPPipeline()

            expected_genres = [
                'terror', 'romance', 'thriller', 'ficção científica',
                'fantasia', 'autoajuda', 'história', 'filosofia',
                'biografia', 'mistério', 'literatura brasileira'
            ]

            for genre in expected_genres:
                assert genre in pipeline.genre_synonyms
                assert isinstance(pipeline.genre_synonyms[genre], list)
                assert len(pipeline.genre_synonyms[genre]) > 0

        def test_famous_authors_structure(self):
            """Teste estrutura dos autores famosos"""
            pipeline = NLPPipeline()

            # Testa alguns autores específicos
            assert 'stephen king' in pipeline.famous_authors
            assert 'dan brown' in pipeline.famous_authors
            assert 'j.k. rowling' in pipeline.famous_authors

            # Testa estrutura das variações
            for author, variations in pipeline.famous_authors.items():
                assert isinstance(variations, list)
                assert len(variations) > 0

        def test_default_configuration(self):
            """Teste configuração padrão"""
            pipeline = NLPPipeline()

            assert pipeline.confidence_threshold == 0.3
            assert pipeline.advanced_processing_enabled is True
