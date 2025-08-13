from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any
import uvicorn

from models.books import Book, BookResponse
from models.chat import ChatRequest, ChatResponse
from services.book_service import BookService
from services.nlp_pipeline import NLPPipeline
from services.recommendation_engine import RecommendationEngine

app = FastAPI(
    title="Bookstore AI Assistant",
    description="IA Híbrida para assistente de livraria virtual",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialização dos serviços
book_service = BookService()
nlp_pipeline = NLPPipeline()
recommendation_engine = RecommendationEngine(book_service)


class BookstoreAI:
    def __init__(self):
        self.book_service = book_service
        self.nlp_pipeline = nlp_pipeline
        self.recommendation_engine = recommendation_engine

        # Estatísticas para monitoramento
        self.stats = {
            'total_requests': 0,
            'simple_processing': 0,
            'advanced_processing': 0,
            'unknown_intents': 0
        }

    def _convert_books_to_dict(self, books: List[Book]) -> List[Dict[str, Any]]:
        """Converte lista de livros para dicionários com compatibilidade Pydantic v1/v2"""
        result = []
        for book in books:
            try:
                # Tenta Pydantic v2 primeiro
                result.append(book.model_dump())
            except AttributeError:
                # Fallback para Pydantic v1
                result.append(book.dict())
        return result

    def process_chat_message(self, message: str) -> ChatResponse:
        """Processa mensagem do chat e retorna resposta"""
        self.stats['total_requests'] += 1

        # Processa com o pipeline
        intent = self.nlp_pipeline.process(message)

        # Atualiza estatísticas baseado no tipo de processamento usado
        if hasattr(self.nlp_pipeline, 'confidence_threshold'):
            if intent.confidence >= self.nlp_pipeline.confidence_threshold:
                self.stats['simple_processing'] += 1
            else:
                self.stats['advanced_processing'] += 1
        else:
            self.stats['simple_processing'] += 1

        if intent.name == 'unknown':
            self.stats['unknown_intents'] += 1

        response_text = ""
        recommended_books: List[Dict[str, Any]] = []

        # Processamento das intenções
        if intent.name == 'recommend_by_genre':
            genre = intent.entities.get('target', '')
            books = self.recommendation_engine.recommend_by_genre(genre)
            recommended_books = self._convert_books_to_dict(books)

            if books:
                response_text = f"Aqui estão algumas recomendações de livros de {genre}:"
            else:
                response_text = f"Desculpe, não encontrei livros do gênero {genre} em nossa base de dados."

        elif intent.name == 'recommend_by_author':
            author = intent.entities.get('author', '')
            books = self.recommendation_engine.recommend_by_author(author)
            recommended_books = self._convert_books_to_dict(books)

            if books:
                response_text = f"Aqui estão os livros de {author} que temos:"
            else:
                response_text = f"Desculpe, não encontrei livros do autor {author} em nossa base de dados."

        elif intent.name == 'recommend_similar':
            book_title = intent.entities.get('target', '')
            books = self.recommendation_engine.recommend_similar_books(book_title)
            recommended_books = self._convert_books_to_dict(books)

            if books:
                response_text = f"Baseado no seu gosto por '{book_title}', recomendo estes livros:"
            else:
                response_text = f"Desculpe, não encontrei o livro '{book_title}' ou livros similares."

        elif intent.name == 'books_by_year':
            year = intent.entities.get('year', 0)
            books = self.recommendation_engine.recommend_by_year(year)
            recommended_books = self._convert_books_to_dict(books)

            if books:
                response_text = f"Aqui estão os livros de {year} que temos:"
            else:
                response_text = f"Desculpe, não encontrei livros do ano {year}."

        elif intent.name == 'bestsellers':
            books = self.recommendation_engine.recommend_bestsellers(True)
            recommended_books = self._convert_books_to_dict(books)
            response_text = "Aqui estão nossos bestsellers:"

        elif intent.name == 'non_bestsellers':
            books = self.recommendation_engine.recommend_bestsellers(False)
            recommended_books = self._convert_books_to_dict(books)
            response_text = "Aqui estão alguns livros menos conhecidos mas interessantes:"

        # ✅ NOVA FUNCIONALIDADE: Recomendação por estado emocional
        elif intent.name == 'recommend_by_mood':
            mood = intent.entities.get('mood', '')
            target_genre = intent.entities.get('target', 'Romance')

            # Verifica se o método existe no recommendation_engine
            if hasattr(self.recommendation_engine, 'recommend_by_mood'):
                books = self.recommendation_engine.recommend_by_mood(mood, target_genre)
                recommended_books = self._convert_books_to_dict(books)

                if books:
                    # ✅ Resposta mais empática e personalizada
                    mood_responses = {
                        'triste': f"Entendo que você está se sentindo triste. Aqui estão alguns livros que podem te ajudar a se sentir melhor:",
                        'deprimido': f"Sei que momentos difíceis passam. Estes livros podem trazer um pouco de luz:",
                        'estressado': f"Para relaxar e esquecer o estresse, recomendo estas leituras:",
                        'ansioso': f"Para acalmar a mente, aqui estão algumas sugestões reconfortantes:",
                        'feliz': f"Que bom que você está feliz! Vamos manter esse astral com estas leituras:",
                        'animado': f"Adorei seu entusiasmo! Aqui estão livros emocionantes para você:",
                    }

                    response_text = mood_responses.get(mood.lower(),
                                                       f"Baseado no seu estado '{mood}', aqui estão algumas sugestões que podem te interessar:")
                else:
                    response_text = f"Desculpe, não encontrei livros adequados para o seu estado atual. Que tal tentar 'livros de romance' ou 'bestsellers'?"
            else:
                response_text = "Funcionalidade de recomendação por humor ainda não implementada."

        # ✅ NOVA FUNCIONALIDADE: Recomendação por ocasião
        elif intent.name == 'recommend_by_occasion':
            occasion = intent.entities.get('occasion', '')
            target_genre = intent.entities.get('target', 'Romance')

            if hasattr(self.recommendation_engine, 'recommend_by_occasion'):
                books = self.recommendation_engine.recommend_by_occasion(occasion, target_genre)
                recommended_books = self._convert_books_to_dict(books)

                if books:
                    # ✅ Respostas contextuais para diferentes ocasiões
                    occasion_responses = {
                        'viajar': f"Perfeito para sua viagem! Aqui estão livros que vão tornar o trajeto mais interessante:",
                        'praia': f"Ótima escolha para relaxar na praia! Estas leituras são perfeitas para o sol e mar:",
                        'férias': f"Férias merecem boas leituras! Aqui estão sugestões para aproveitar seu tempo livre:",
                        'trabalho': f"Para ler nos intervalos do trabalho, aqui estão algumas opções:",
                        'dormir': f"Para uma leitura relaxante antes de dormir:",
                        'avião': f"Para tornar o voo mais agradável, recomendo:",
                    }

                    response_text = occasion_responses.get(occasion.lower(),
                                                           f"Para a ocasião '{occasion}', aqui estão minhas sugestões:")
                else:
                    response_text = f"Desculpe, não encontrei livros adequados para '{occasion}'. Que tal tentar uma busca mais específica?"
            else:
                response_text = "Funcionalidade de recomendação por ocasião ainda não implementada."

        else:
            response_text = self._generate_helpful_response(message, intent.confidence)

        return ChatResponse(
            response=response_text,
            recommended_books=recommended_books,
            intent=intent.name,
            confidence=intent.confidence
        )

    def _generate_helpful_response(self, message: str, confidence: float) -> str:
        """Gera resposta mais útil para intenções desconhecidas"""
        if confidence > 0.1:
            return ("Entendi parcialmente sua pergunta, mas preciso de mais detalhes. "
                    "Você pode me perguntar sobre livros por gênero, autor, ano, "
                    "seu estado emocional, ou pedir recomendações baseadas em livros que você gostou. "
                    "Por exemplo: 'Estou triste, preciso de algo que me anime' ou 'Vou viajar, que livro levar?'")
        else:
            return ("Desculpe, não entendi sua pergunta. Você pode tentar perguntar de outra forma? "
                    "Exemplos do que posso ajudar:\n"
                    "• 'Gosto de livros de terror'\n"
                    "• 'Me mostre livros do Stephen King'\n"
                    "• 'Estou me sentindo triste hoje'\n"
                    "• 'Vou viajar, que livro você recomenda?'\n"
                    "• 'Li Inferno e gostei, que outros você sugere?'")

    # ✅ NOVA FUNCIONALIDADE: Análise de sentimento da conversa
    def _analyze_conversation_sentiment(self, message: str) -> str:
        """Analisa o sentimento da mensagem para personalizar a resposta"""
        positive_words = ['gosto', 'adoro', 'amo', 'feliz', 'animado', 'ótimo', 'excelente']
        negative_words = ['triste', 'deprimido', 'estressado', 'ansioso', 'ruim', 'mal']

        message_lower = message.lower()

        positive_count = sum(1 for word in positive_words if word in message_lower)
        negative_count = sum(1 for word in negative_words if word in message_lower)

        if negative_count > positive_count:
            return 'negative'
        elif positive_count > negative_count:
            return 'positive'
        else:
            return 'neutral'

    # ✅ FUNCIONALIDADE MELHORADA: Estatísticas mais detalhadas
    def get_ai_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas da IA para monitoramento"""
        total = max(self.stats['total_requests'], 1)

        stats_response: Dict[str, Any] = {
            **self.stats,
            'simple_processing_percentage': round((self.stats['simple_processing'] / total) * 100, 2),
            'advanced_processing_percentage': round((self.stats['advanced_processing'] / total) * 100, 2),
            'unknown_intents_percentage': round((self.stats['unknown_intents'] / total) * 100, 2),
            'success_rate': round(((total - self.stats['unknown_intents']) / total) * 100, 2),
        }

        if hasattr(self.nlp_pipeline, 'confidence_threshold'):
            stats_response['nlp_settings'] = {
                'confidence_threshold': self.nlp_pipeline.confidence_threshold,
                'advanced_processing_enabled': getattr(self.nlp_pipeline, 'advanced_processing_enabled', False)
            }

        return stats_response

    # ✅ NOVA FUNCIONALIDADE: Resetar estatísticas
    def reset_stats(self) -> Dict[str, str]:
        """Reseta as estatísticas da IA"""
        self.stats = {
            'total_requests': 0,
            'simple_processing': 0,
            'advanced_processing': 0,
            'unknown_intents': 0
        }
        return {"message": "Estatísticas resetadas com sucesso"}

    def configure_nlp(self, confidence_threshold: Optional[float] = None,
                      advanced_processing: Optional[bool] = None) -> None:
        """Permite configurar o NLP em runtime"""
        if hasattr(self.nlp_pipeline, 'set_confidence_threshold') and confidence_threshold is not None:
            self.nlp_pipeline.set_confidence_threshold(confidence_threshold)

        if hasattr(self.nlp_pipeline, 'set_advanced_processing') and advanced_processing is not None:
            self.nlp_pipeline.set_advanced_processing(advanced_processing)


# Instância da IA
bookstore_ai = BookstoreAI()


# Endpoints da API
@app.get("/")
async def root():
    return {"message": "Bookstore AI Assistant API", "version": "1.0.0"}


@app.get("/books", response_model=BookResponse)
async def get_all_books():
    """Retorna todos os livros disponíveis"""
    books = book_service.get_all_books()
    return BookResponse(books=books, total=len(books))


@app.get("/books/{book_id}", response_model=Book)
async def get_book_by_id(book_id: int):
    """Retorna um livro específico por ID"""
    book = book_service.get_book_by_id(book_id)
    if not book:
        raise HTTPException(status_code=404, detail="Livro não encontrado")
    return book


@app.get("/books/genre/{genre}", response_model=BookResponse)
async def get_books_by_genre(genre: str):
    """Retorna livros por gênero"""
    books = book_service.get_books_by_genre(genre)
    return BookResponse(books=books, total=len(books))


@app.get("/books/author/{author}", response_model=BookResponse)
async def get_books_by_author(author: str):
    """Retorna livros por autor"""
    books = book_service.get_books_by_author(author)
    return BookResponse(books=books, total=len(books))


@app.post("/chat", response_model=ChatResponse)
async def chat_with_ai(request: ChatRequest):
    """Endpoint principal do chat com a IA"""
    try:
        response = bookstore_ai.process_chat_message(request.message)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")


@app.get("/health")
async def health_check():
    """Endpoint de verificação de saúde da API"""
    return {
        "status": "healthy",
        "total_books": len(book_service.get_all_books()),
        "services": {
            "book_service": "active",
            "nlp_pipeline": "active",
            "recommendation_engine": "active"
        }
    }


@app.get("/ai/stats")
async def get_ai_statistics():
    """Retorna estatísticas de uso da IA"""
    return bookstore_ai.get_ai_stats()


@app.post("/ai/configure")
async def configure_ai(
        confidence_threshold: Optional[float] = None,
        advanced_processing: Optional[bool] = None
):
    """Permite configurar parâmetros da IA em runtime"""
    try:
        bookstore_ai.configure_nlp(confidence_threshold, advanced_processing)

        current_settings: Dict[str, Any] = {"message": "Configuração atualizada com sucesso"}

        # Adiciona configurações atuais se disponíveis
        if hasattr(bookstore_ai.nlp_pipeline, 'confidence_threshold'):
            current_settings["current_settings"] = {
                "confidence_threshold": bookstore_ai.nlp_pipeline.confidence_threshold,
                "advanced_processing_enabled": getattr(bookstore_ai.nlp_pipeline, 'advanced_processing_enabled', False)
            }

        return current_settings
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro na configuração: {str(e)}")


@app.get("/ai/test-processing/{message}")
async def test_processing_modes(message: str):
    """Endpoint para testar diferentes modos de processamento"""
    try:
        # Resultado híbrido (sempre funciona)
        hybrid_result = bookstore_ai.nlp_pipeline.process(message)

        response: Dict[str, Any] = {
            "message": message,
            "hybrid_result": {
                "intent": hybrid_result.name,
                "confidence": hybrid_result.confidence,
                "entities": hybrid_result.entities
            }
        }

        # Testa processamento simples se disponível
        if hasattr(bookstore_ai.nlp_pipeline, '_process_with_regex'):
            simple_result = bookstore_ai.nlp_pipeline._process_with_regex(message)
            response["simple_processing"] = {
                "intent": simple_result.name,
                "confidence": simple_result.confidence,
                "entities": simple_result.entities
            }

        # Testa processamento avançado se disponível
        if hasattr(bookstore_ai.nlp_pipeline, '_process_with_advanced_nlp'):
            advanced_result = bookstore_ai.nlp_pipeline._process_with_advanced_nlp(message)
            response["advanced_processing"] = {
                "intent": advanced_result.name,
                "confidence": advanced_result.confidence,
                "entities": advanced_result.entities
            }

        # Determina qual processamento foi usado
        if hasattr(bookstore_ai.nlp_pipeline, 'confidence_threshold'):
            response[
                "processing_used"] = "simple" if hybrid_result.confidence >= bookstore_ai.nlp_pipeline.confidence_threshold else "advanced"
        else:
            response["processing_used"] = "standard"

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no teste: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)