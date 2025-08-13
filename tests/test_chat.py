import pytest
from pydantic import ValidationError
from typing import List, Dict, Any

from models.chat import ChatRequest, ChatResponse

class TestChatRequest:
    """Testes para o modelo ChatRequest"""

    def test_chat_request_creation_valid_data(self):
        """Teste criação de ChatRequest com dados válidos"""
        request = ChatRequest(
            message="Gosto de livros de terror",
            user_id="user123"
        )

        assert request.message == "Gosto de livros de terror"
        assert request.user_id == "user123"

    def test_chat_request_creation_without_user_id(self):
        """Teste criação de ChatRequest sem user_id (opcional)"""
        request = ChatRequest(message="Quero livros de romance")

        assert request.message == "Quero livros de romance"
        assert request.user_id is None

    def test_chat_request_creation_with_none_user_id(self):
        """Teste criação de ChatRequest com user_id explicitamente None"""
        request = ChatRequest(
            message="Me recomende algo",
            user_id=None
        )

        assert request.message == "Me recomende algo"
        assert request.user_id is None

    def test_chat_request_creation_empty_message(self):
        """Teste criação de ChatRequest com mensagem vazia"""
        request = ChatRequest(message="", user_id="user123")

        assert request.message == ""
        assert request.user_id == "user123"

    def test_chat_request_creation_whitespace_message(self):
        """Teste criação de ChatRequest com mensagem só com espaços"""
        request = ChatRequest(message="   ", user_id="user123")

        assert request.message == "   "
        assert request.user_id == "user123"

    @pytest.mark.parametrize("message,user_id", [
        ("Mensagem simples", "user1"),
        ("Mensagem com números 123", "user_456"),
        ("Mensagem com símbolos !@#$%", "user-test"),
        ("Mensagem muito longa " * 100, "very_long_user_id_" * 10),
        ("Mensagem com acentos: ção, ã, é", "usuário_123"),
        ("Message in English", "english_user"),
        ("Mensaje en español", "usuario_español"),
        ("🤖 Mensagem com emoji 📚", "emoji_user_😊"),
    ])
    def test_chat_request_various_valid_inputs(self, message, user_id):
        """Teste ChatRequest com várias entradas válidas"""
        request = ChatRequest(message=message, user_id=user_id)

        assert request.message == message
        assert request.user_id == user_id

    @pytest.mark.parametrize("invalid_message,expected_error", [
        (None, "Input should be a valid string"),
        (123, "Input should be a valid string"),
        ([], "Input should be a valid string"),
        ({}, "Input should be a valid string"),
        (True, "Input should be a valid string"),
    ])
    def test_chat_request_invalid_message_types(self, invalid_message, expected_error):
        """Teste ChatRequest com tipos inválidos para message"""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message=invalid_message, user_id="user123")

        assert expected_error in str(exc_info.value)

    @pytest.mark.parametrize("invalid_user_id,expected_error", [
        (123, "Input should be a valid string"),
        ([], "Input should be a valid string"),
        ({}, "Input should be a valid string"),
        (True, "Input should be a valid string"),
    ])
    def test_chat_request_invalid_user_id_types(self, invalid_user_id, expected_error):
        """Teste ChatRequest com tipos inválidos para user_id"""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(message="Valid message", user_id=invalid_user_id)

        assert expected_error in str(exc_info.value)

    def test_chat_request_missing_required_message(self):
        """Teste ChatRequest sem campo obrigatório message"""
        with pytest.raises(ValidationError) as exc_info:
            ChatRequest(user_id="user123")

        assert "Field required" in str(exc_info.value)
        assert "message" in str(exc_info.value)

    def test_chat_request_extra_fields_ignored(self):
        """Teste que campos extras são ignorados"""
        request = ChatRequest(
            message="Test message",
            user_id="user123",
            extra_field="should_be_ignored",
            another_extra=456
        )

        assert request.message == "Test message"
        assert request.user_id == "user123"
        assert not hasattr(request, "extra_field")
        assert not hasattr(request, "another_extra")

    def test_chat_request_model_dump(self):
        """Teste serialização do modelo ChatRequest"""
        request = ChatRequest(
            message="Test message",
            user_id="user123"
        )

        request_dict = request.model_dump()

        assert isinstance(request_dict, dict)
        assert request_dict["message"] == "Test message"
        assert request_dict["user_id"] == "user123"
        assert len(request_dict) == 2

    def test_chat_request_model_dump_without_user_id(self):
        """Teste serialização do ChatRequest sem user_id"""
        request = ChatRequest(message="Test message")

        request_dict = request.model_dump()

        assert request_dict["message"] == "Test message"
        assert request_dict["user_id"] is None

    def test_chat_request_model_dump_json(self):
        """Teste serialização JSON do modelo ChatRequest"""
        request = ChatRequest(
            message="Test message",
            user_id="user123"
        )

        request_json = request.model_dump_json()

        assert isinstance(request_json, str)
        assert '"message":"Test message"' in request_json or '"message": "Test message"' in request_json
        assert '"user_id":"user123"' in request_json or '"user_id": "user123"' in request_json

    def test_chat_request_equality(self):
        """Teste igualdade entre objetos ChatRequest"""
        request1 = ChatRequest(message="Same message", user_id="same_user")
        request2 = ChatRequest(message="Same message", user_id="same_user")
        request3 = ChatRequest(message="Different message", user_id="same_user")
        request4 = ChatRequest(message="Same message", user_id="different_user")

        assert request1 == request2
        assert request1 != request3
        assert request1 != request4
        assert request1 is not request2  # Objetos diferentes

    def test_chat_request_hash(self):
        """Teste hash de objetos ChatRequest"""
        request1 = ChatRequest(message="Test", user_id="user1")
        request2 = ChatRequest(message="Test", user_id="user1")

        # Objetos iguais devem ter o mesmo hash
        assert hash(request1) == hash(request2)

        # Pode ser usado em sets
        request_set = {request1, request2}
        assert len(request_set) == 1  # Apenas um objeto único

    def test_chat_request_string_representation(self):
        """Teste representação string do ChatRequest"""
        request = ChatRequest(message="Test message", user_id="user123")

        request_str = str(request)
        request_repr = repr(request)

        assert "Test message" in request_str or "Test message" in request_repr
        assert "ChatRequest" in request_repr

