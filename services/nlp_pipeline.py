import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from difflib import SequenceMatcher

@dataclass
class Intent:
    name: str
    confidence: float
    entities: Dict[str, str]


class NLPPipeline:
    def __init__(self):
        # Padrões regex originais (processamento rápido)
        self.intent_patterns = {
            'recommend_by_genre': [
                r'gosto de livros de (\w+)',
                r'recomend[ae] livros de (\w+)',
                r'livros do g[êe]nero (\w+)',
                r'quero ler (\w+)',
                r'me interessam? (\w+)',
                r'curto (\w+)',
                r'adoro (\w+)',
                # Padrões para gêneros compostos
                r'gosto de (ficção científica|literatura brasileira)',
                r'quero ler (ficção científica|literatura brasileira)',
                r'livros de (ficção científica|literatura brasileira)',
                r'recomend[ae] (ficção científica|literatura brasileira)',
                r'algo de (\w+)',
                r'livros (\w+)',
                r'estou procurando (\w+)',
                r'queria (\w+)',
                r'busco (\w+)',
                r'preciso de (\w+)',
            ],
            'recommend_by_author': [
                r'livros do autor (.+?)(?:\s|$)',
                r'obras de (.+?)(?:\s|$)',
                r'mostre livros de (.+?)(?:\s|$)',
                r'autor (.+?)(?:\s|$)',
                r'escrito por (.+?)(?:\s|$)',
                r'(.+?) escreveu',
                r'tem (.+?) na biblioteca',
                r'conhece (.+?)?',
                r'já leu (.+?)?',
            ],
            'recommend_similar': [
                r'li o livro (.+?) e gostei',
                r'gostei do livro (.+?)',
                r'similar ao (.+?)',
                r'parecido com (.+?)',
                r'baseado em (.+?)',
                r'como (.+?)',
                r'estilo (.+?)',
                r'no mesmo tom de (.+?)',
                r'que lembre (.+?)',
            ],
            'books_by_year': [
                r'livros de (\d{4})',
                r'publicados em (\d{4})',
                r'do ano (\d{4})',
                r'lan[çc]ados em (\d{4})',
                r'dos anos (\d{4})',
                r'da década de (\d{4})',
                r'século (\d{1,2})',
            ],
            'bestsellers': [
                r'bestsellers?',
                r'mais vendidos',
                r'livros famosos',
                r'populares',
                r'sucessos',
                r'consagrados',
                r'aclamados',
                r'premiados',
            ],
            'non_bestsellers': [
                r'n[ãa]o bestsellers?',
                r'menos conhecidos',
                r'livros raros',
                r'indies?',
                r'independentes',
                r'cult',
                r'underground',
                r'nicho',
            ],
            'recommend_by_mood': [
                r'estou (me sentindo )?(?:muito )?(triste|feliz|estressado|ansioso|deprimido|animado|bem|mal)',
                r'me sentindo (triste|feliz|estressado|ansioso|deprimido|animado|bem|mal|para baixo)',
                r'preciso de algo (?:que me )?(anime|relaxe|emocione|inspire|divirta)',
                r'quero (chorar|rir|me emocionar|relaxar|me inspirar)',
                r'algo (emocionante|relaxante|triste|divertido|inspirador|leve|pesado)',
                r'humor (triste|feliz|estressado|ansioso|animado)',
                r'dia (ruim|bom|difícil|estressante)',
                r'preciso (relaxar|me animar|chorar|rir)',
                r'estou (nostálgico|melancólico|reflexivo|contemplativo)',
                r'preciso (refletir|pensar|filosofar|meditar)',
                r'quero algo (profundo|superficial|intelectual|simples)',
            ],
            'recommend_by_occasion': [
                r'vou (viajar|para a praia|de férias|trabalhar)',
                r'para ler (na praia|no avião|antes de dormir|no trabalho)',
                r'algo para (relaxar|passar o tempo|viagem|férias)',
                r'leitura (leve|rápida|para viagem|de férias)',
                r'para (estudar|aprender|crescer|evoluir)',
                r'no (metrô|ônibus|trem|transporte)',
                r'entre (aulas|reuniões|compromissos)',
            ]
        }

        # Mapeamento de humor para todos os gêneros disponíveis
        self.mood_to_genre = {
            # Estados negativos → Gêneros que elevam o ânimo
            'triste': 'Romance',
            'deprimido': 'Autoajuda',
            'mal': 'Romance',
            'para baixo': 'Autoajuda',
            'desanimado': 'Autoajuda',
            'perdido': 'Filosofia',
            'confuso': 'Filosofia',
            'cansado': 'Romance',
            'nostálgico': 'Romance',
            'melancólico': 'Romance',

            # Estados positivos → Gêneros que mantêm energia
            'feliz': 'Fantasia',
            'animado': 'Thriller',
            'bem': 'História',
            'motivado': 'Biografia',
            'empolgado': 'Fantasia',
            'confiante': 'Thriller',
            'determinado': 'Biografia',

            # Estados reflexivos → Gêneros profundos
            'reflexivo': 'Filosofia',
            'contemplativo': 'Filosofia',

            # Ações desejadas
            'anime': 'Romance',
            'relaxe': 'Romance',
            'emocione': 'Romance',
            'inspire': 'Biografia',
            'divirta': 'Fantasia',
            'chorar': 'Romance',
            'rir': 'Romance',
            'relaxar': 'Romance',
            'refletir': 'Filosofia',
            'pensar': 'Filosofia',
            'filosofar': 'Filosofia',
            'meditar': 'Filosofia',

            # Características desejadas
            'emocionante': 'Thriller',
            'relaxante': 'Romance',
            'divertido': 'Fantasia',
            'inspirador': 'Biografia',
            'leve': 'Romance',
            'pesado': 'Filosofia',
            'profundo': 'Filosofia',
            'superficial': 'Romance',
            'intelectual': 'Filosofia',
            'simples': 'Romance',
        }

        # Mapeamento de ocasiões para gêneros apropriados
        self.occasion_to_genre = {
            # Lazer e descanso
            'viajar': 'Romance',
            'praia': 'Romance',
            'férias': 'Fantasia',
            'fim de semana': 'Fantasia',
            'feriado': 'Fantasia',
            'tempo livre': 'Romance',
            'relaxar': 'Romance',

            # Trabalho e desenvolvimento
            'trabalhar': 'Autoajuda',
            'trabalho': 'Autoajuda',
            'estudar': 'História',
            'aprender': 'História',
            'crescer': 'Autoajuda',
            'evoluir': 'Biografia',

            # Transporte
            'avião': 'Mistério',
            'metrô': 'Thriller',
            'ônibus': 'Romance',
            'trem': 'Mistério',
            'transporte': 'Romance',

            # Intervalos
            'dormir': 'Romance',
            'aulas': 'Filosofia',
            'reuniões': 'Autoajuda',
            'compromissos': 'Romance',

            # Contextos específicos
            'viagem': 'Mistério',
        }

        # Sinônimos para todos os gêneros
        self.genre_synonyms = {
            'terror': ['horror', 'medo', 'assombração', 'suspense', 'assustador', 'macabro', 'sombrio', 'gótico'],
            'romance': ['amor', 'romântico', 'paixão', 'amoroso', 'sentimental', 'emocional', 'tocante', 'dramático'],
            'thriller': ['suspense', 'ação', 'tensão', 'mistério', 'emocionante', 'adrenalina', 'investigação'],
            'ficção científica': ['sci-fi', 'futurista', 'espacial', 'ficção', 'científica', 'futurismo', 'distopia',
                                  'cyberpunk'],
            'fantasia': ['magia', 'épico', 'medieval', 'mágico', 'fantástico', 'aventura', 'mitológico', 'lendário'],
            'autoajuda': ['auto-ajuda', 'desenvolvimento', 'pessoal', 'motivação', 'sucesso', 'crescimento', 'coaching',
                          'liderança'],
            'história': ['histórico', 'passado', 'civilização', 'humanidade', 'histórica', 'época', 'guerra',
                         'biografia histórica'],
            'filosofia': ['filosófico', 'reflexão', 'pensamento', 'existencial', 'reflexivo', 'profundo', 'intelectual',
                          'contemplativo'],
            'biografia': ['biográfico', 'vida', 'personalidade', 'famoso', 'autobiografia', 'memórias', 'trajetória',
                          'história de vida'],
            'mistério': ['misterioso', 'enigma', 'detetive', 'investigação', 'crime', 'policial', 'suspense', 'noir'],
            'literatura brasileira': ['brasileiro', 'brasil', 'nacional', 'literatura', 'brasiliense', 'tupiniquim',
                                      'verde-amarelo'],
        }

        # Mapeamento de autores para facilitar reconhecimento
        self.famous_authors = {
            # Nomes completos e variações
            'stephen king': ['king', 'stephen', 'steve king'],
            'dan brown': ['brown', 'dan', 'daniel brown'],
            'j.k. rowling': ['rowling', 'jk rowling', 'joanne rowling'],
            'j.r.r. tolkien': ['tolkien', 'jrr tolkien', 'john tolkien'],
            'agatha christie': ['christie', 'agatha'],
            'arthur conan doyle': ['doyle', 'conan doyle', 'arthur doyle'],
            'george orwell': ['orwell', 'george'],
            'isaac asimov': ['asimov', 'isaac'],
            'frank herbert': ['herbert', 'frank'],
            'h.g. wells': ['wells', 'hg wells', 'herbert wells'],
            'edgar allan poe': ['poe', 'edgar poe', 'allan poe'],
            'franz kafka': ['kafka', 'franz'],
            'albert camus': ['camus', 'albert'],
            'gabriel garcía márquez': ['márquez', 'garcia marquez', 'gabriel garcia'],
            'machado de assis': ['machado', 'assis', 'machado assis'],
            'paulo coelho': ['coelho', 'paulo'],
            'yuval noah harari': ['harari', 'yuval', 'noah harari'],
        }

        # Threshold para decidir quando usar processamento avançado
        self.confidence_threshold = 0.3

        # Flag para habilitar processamento avançado
        self.advanced_processing_enabled = True

    def process(self, text: str) -> Intent:
        """Processa o texto e identifica a intenção"""
        text = text.lower().strip()

        # Aplica expansão de sinônimos ANTES do processamento
        text = self._expand_synonyms(text)

        for intent_name, patterns in self.intent_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text, re.IGNORECASE)
                if match:
                    entities = {}
                    confidence = self._calculate_confidence(text, pattern)

                    if match.groups():
                        if intent_name in ['recommend_by_genre', 'recommend_similar']:
                            entities['target'] = match.group(1).strip()
                        elif intent_name == 'recommend_by_author':
                            author_name = match.group(1).strip()
                            # Normaliza nome do autor extraído
                            normalized_author = self._find_best_author_match(author_name)
                            entities['author'] = normalized_author if normalized_author else author_name
                        elif intent_name == 'books_by_year':
                            year_str = match.group(1)
                            # Tratamento especial para séculos
                            if 'século' in text:
                                century = int(year_str)
                                # Converte século para ano aproximado
                                entities['year'] = (century - 1) * 100 + 50  # Meio do século
                            else:
                                entities['year'] = int(year_str)
                        elif intent_name == 'recommend_by_mood':
                            # Processa estado emocional
                            mood_word = match.group(1).strip() if match.group(1) else match.group(0).strip()
                            target_genre = self.mood_to_genre.get(mood_word, 'Romance')
                            entities['mood'] = mood_word
                            entities['target'] = target_genre
                        elif intent_name == 'recommend_by_occasion':
                            # Processa ocasião
                            occasion_word = match.group(1).strip() if match.group(1) else match.group(0).strip()
                            target_genre = self.occasion_to_genre.get(occasion_word, 'Romance')
                            entities['occasion'] = occasion_word
                            entities['target'] = target_genre

                    return Intent(intent_name, confidence, entities)

        return Intent('unknown', 0.0, {})

    # Normaliza nomes de autores no texto
    def _normalize_author_names(self, text: str) -> str:
        """Normaliza variações de nomes de autores famosos"""
        for full_name, variations in self.famous_authors.items():
            for variation in variations:
                # Substitui variações pelo nome completo
                pattern = r'\b' + re.escape(variation) + r'\b'
                text = re.sub(pattern, full_name, text, flags=re.IGNORECASE)
        return text

    # Encontra melhor correspondência de autor
    def _find_best_author_match(self, author_input: str) -> Optional[str]:
        """Encontra a melhor correspondência para um nome de autor"""
        author_lower = author_input.lower().strip()

        # Busca exata primeiro
        if author_lower in self.famous_authors:
            return author_lower

        # Busca por variações
        for full_name, variations in self.famous_authors.items():
            if author_lower in variations:
                return full_name

        # Busca por similaridade usando SequenceMatcher
        best_match = None
        best_ratio = 0.0

        all_authors = list(self.famous_authors.keys())
        for author in all_authors:
            ratio = SequenceMatcher(None, author_lower, author).ratio()
            if ratio > best_ratio and ratio > 0.7:  # Threshold de similaridade
                best_ratio = ratio
                best_match = author

        return best_match

    def _expand_synonyms(self, text: str) -> str:
        """Expande sinônimos para melhorar correspondência"""
        # Processa sinônimos de forma mais inteligente
        for main_genre, synonyms in self.genre_synonyms.items():
            for synonym in synonyms:
                # Usa word boundaries para evitar substituições parciais
                pattern = r'\b' + re.escape(synonym) + r'\b'
                text = re.sub(pattern, main_genre, text, flags=re.IGNORECASE)

        return text

    def _calculate_confidence(self, text: str, pattern: str) -> float:
        """Calcula a confiança baseada na correspondência do padrão"""
        try:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                match_length = len(match.group(0))
                text_length = len(text)
                base_confidence = min(0.95, (match_length / text_length) * 2)

                # Bonus se o match está no início da frase
                if match.start() < len(text) * 0.3:
                    base_confidence += 0.1

                return min(base_confidence, 0.95)
        except:
            pass

        return 0.3

        # Keywords atualizadas com todos os 100 livros

    def _calculate_keyword_scores(self, words: List[str]) -> Dict[str, float]:
        """Calcula pontuações baseadas em palavras-chave"""
        keyword_weights = {
            'recommend_by_genre': {
                'gosto': 0.3, 'adoro': 0.3, 'curto': 0.2, 'interesse': 0.2,
                'terror': 0.4, 'romance': 0.4, 'thriller': 0.4, 'ficção': 0.4,
                'fantasia': 0.4, 'autoajuda': 0.4, 'história': 0.4, 'filosofia': 0.4,
                'biografia': 0.4, 'mistério': 0.4, 'literatura': 0.4,
                'gênero': 0.3, 'tipo': 0.2, 'categoria': 0.2, 'estilo': 0.2
            },
            'recommend_by_author': {
                'autor': 0.5, 'escritor': 0.4, 'escrito': 0.3, 'obra': 0.3,
                'brown': 0.3, 'king': 0.3, 'orwell': 0.3, 'austen': 0.3,
                'green': 0.3, 'harari': 0.3, 'rowling': 0.3, 'coelho': 0.3,
                'tolkien': 0.3, 'christie': 0.3, 'assis': 0.3, 'isaacson': 0.3,
                'lewis': 0.3, 'flynn': 0.3, 'harris': 0.3, 'stoker': 0.3,
                'shelley': 0.3, 'asimov': 0.3, 'huxley': 0.3, 'hill': 0.3,
                'duhigg': 0.3, 'moyes': 0.3, 'hawking': 0.3, 'diamond': 0.3,
                'camus': 0.3, 'nietzsche': 0.3, 'amado': 0.3, 'azevedo': 0.3,
                'malala': 0.3, 'gibson': 0.3, 'herbert': 0.3, 'gaarder': 0.3,
                'marco': 0.3, 'aurélio': 0.3, 'covey': 0.3, 'carnegie': 0.3,
                'tolle': 0.3, 'dweck': 0.3, 'doyle': 0.3, 'blatty': 0.3,
                'eco': 0.3, 'zusak': 0.3, 'poe': 0.3, 'tzu': 0.3, 'dick': 0.3,
                'saint-exupéry': 0.3, 'johnson': 0.3, 'freyre': 0.3, 'alencar': 0.3,
                'sampson': 0.3, 'kafka': 0.3, 'márquez': 0.3, 'collins': 0.3,
                'roth': 0.3, 'toro': 0.3, 'zafón': 0.3, 'hosseini': 0.3,
                'young': 0.3, 'boyne': 0.3, 'bloch': 0.3, 'lovecraft': 0.3,
                'matheson': 0.3, 'stevenson': 0.3, 'salinger': 0.3, 'bradbury': 0.3,
                'burgess': 0.3, 'clarke': 0.3, 'adams': 0.3, 'wells': 0.3,
                'wilde': 0.3, 'dostoiévski': 0.3, 'tolstói': 0.3, 'fitzgerald': 0.3,
                'melville': 0.3, 'lee': 0.3, 'kahneman': 0.3, 'kiyosaki': 0.3,
                'hunter': 0.3, 'murphy': 0.3
            },
            'recommend_similar': {
                'similar': 0.5, 'parecido': 0.4, 'como': 0.3, 'baseado': 0.3,
                'gostei': 0.4, 'adorei': 0.4, 'li': 0.3, 'terminei': 0.3,
                'igual': 0.4, 'semelhante': 0.4, 'estilo': 0.3, 'tom': 0.3
            },
            'bestsellers': {
                'bestseller': 0.6, 'famoso': 0.4, 'popular': 0.4, 'vendido': 0.3,
                'sucesso': 0.3, 'conhecido': 0.2, 'clássico': 0.4, 'renomado': 0.3,
                'consagrado': 0.4, 'aclamado': 0.3, 'premiado': 0.3
            }
        }

        scores = {}
        for intent, keywords in keyword_weights.items():
            score = 0
            for word in words:
                if word in keywords:
                    score += keywords[word]

            if len(words) > 0:
                scores[intent] = min(score / len(words), 1.0)

        return scores

    # Detecção de títulos específicos dos 100 livros
    def _detect_specific_books(self, text: str) -> List[str]:
        """Detecta menções a títulos específicos dos 100 livros"""
        # Lista dos títulos mais reconhecíveis (simplificados)
        famous_titles = [
            'inferno', 'código da vinci', 'it', 'iluminado', 'cemitério maldito',
            '1984', 'anjos e demônios', 'carrie', 'poder do agora', 'harry potter',
            'alquimista', 'steve jobs', 'mindset', 'duna', 'senhor dos anéis',
            'dom casmurro', 'sapiens', 'mundo de sofia', 'orgulho e preconceito',
            'culpa é das estrelas', 'hobbit', 'nárnia', 'garota exemplar',
            'silêncio dos inocentes', 'drácula', 'frankenstein', 'fundação',
            'admirável mundo novo', 'pequeno príncipe', 'processo', 'metamorfose',
            'estrangeiro', 'jogos vorazes', 'divergente', 'fahrenheit 451',
            'laranja mecânica', 'guerra dos mundos', 'retrato de dorian gray',
            'crime e castigo', 'guerra e paz', 'grande gatsby', 'moby dick'
        ]


        detected_books = []
        text_lower = text.lower()

        for title in famous_titles:
            if title in text_lower:
                detected_books.append(title)

        return detected_books

    # Sugestão inteligente de gêneros
    def _suggest_smart_genre(self, user_input: str) -> Optional[str]:
        """Sugere gênero baseado em contexto e palavras-chave"""
        text_lower = user_input.lower()

        # Mapeamento contextual
        context_mapping = {
            'clássico': 'Literatura Brasileira',
            'antigo': 'Filosofia',
            'moderno': 'Ficção Científica',
            'contemporâneo': 'Romance',
            'jovem': 'Romance',
            'adulto': 'Thriller',
            'criança': 'Fantasia',
            'escola': 'História',
            'trabalho': 'Autoajuda',
            'universidade': 'Filosofia',
            'pesado': 'Filosofia',
            'leve': 'Romance',
            'rápido': 'Thriller',
            'longo': 'História',
            'curto': 'Romance',
            'denso': 'Filosofia',
            'fácil': 'Romance',
            'difícil': 'Filosofia',
        }

        for keyword, genre in context_mapping.items():
            if keyword in text_lower:
                return genre

        return None

    # Análise de sentimento mais sofisticada
    def _analyze_advanced_sentiment(self, text: str) -> Dict[str, float]:
        """Análise de sentimento mais detalhada"""
        positive_words = ['gosto', 'adoro', 'amo', 'feliz', 'animado', 'ótimo', 'excelente', 'maravilhoso', 'incrível']
        negative_words = ['triste', 'deprimido', 'estressado', 'ansioso', 'ruim', 'mal', 'péssimo', 'horrível']
        neutral_words = ['quero', 'preciso', 'busco', 'procuro', 'gostaria', 'desejo']

        text_lower = text.lower()

        positive_score = sum(1 for word in positive_words if word in text_lower)
        negative_score = sum(1 for word in negative_words if word in text_lower)
        neutral_score = sum(1 for word in neutral_words if word in text_lower)

        total = positive_score + negative_score + neutral_score

        if total == 0:
            return {'positive': 0.33, 'negative': 0.33, 'neutral': 0.34}

        return {
            'positive': positive_score / total,
            'negative': negative_score / total,
            'neutral': neutral_score / total
        }

        # Detecção de múltiplos critérios

    def _detect_multiple_criteria(self, text: str) -> Dict[str, any]:
        """Detecta múltiplos critérios em uma única consulta"""
        criteria = {}

        # Detecta gênero
        for genre in ['terror', 'romance', 'thriller', 'ficção científica', 'fantasia',
                      'autoajuda', 'história', 'filosofia', 'biografia', 'mistério', 'literatura brasileira']:
            if genre in text.lower():
                criteria['genre'] = genre
                break

        # Detecta autor
        for author in self.famous_authors.keys():
            if author in text.lower():
                criteria['author'] = author
                break

        # Detecta ano/período
        year_match = re.search(r'\b(19|20)\d{2}\b', text)
        if year_match:
            criteria['year'] = int(year_match.group())

        # Detecta bestseller
        if any(word in text.lower() for word in ['bestseller', 'famoso', 'popular', 'clássico']):
            criteria['bestseller'] = True
        elif any(word in text.lower() for word in ['raro', 'independente', 'alternativo']):
            criteria['bestseller'] = False

        return criteria

        # Sugestões inteligentes para consultas não reconhecidas

    def _generate_suggestions(self, text: str) -> List[str]:
        """Gera sugestões para consultas não reconhecidas"""
        suggestions = []

        # Detecta possíveis gêneros mal escritos
        for genre in self.genre_synonyms.keys():
            ratio = SequenceMatcher(None, text.lower(), genre).ratio()
            if ratio > 0.6:
                suggestions.append(f"Você quis dizer '{genre}'?")

        # Detecta possíveis autores mal escritos
        for author in self.famous_authors.keys():
            ratio = SequenceMatcher(None, text.lower(), author).ratio()
            if ratio > 0.6:
                suggestions.append(f"Você quis dizer '{author}'?")

        return suggestions[:3]  # Máximo 3 sugestões

        # Validação inteligente de entrada

    def _validate_input(self, text: str) -> Dict[str, any]:
        """Valida e analisa a entrada do usuário"""
        analysis = {
            'length': len(text),
            'word_count': len(text.split()),
            'has_question_mark': '?' in text,
            'has_exclamation': '!' in text,
            'language_detected': 'portuguese',  # Simplificado
            'complexity': 'simple' if len(text.split()) <= 5 else 'complex'
        }

        return analysis

    # Método para validar se um gênero existe na base de dados
    def _validate_genre(self, genre: str) -> str:
        """Valida e normaliza o gênero baseado nos livros disponíveis"""
        available_genres = [
            'Thriller', 'Terror', 'Ficção Científica', 'Autoajuda',
            'Romance', 'História', 'Filosofia', 'Fantasia',
            'Literatura Brasileira', 'Biografia', 'Mistério'
        ]

        genre_lower = genre.lower()

        # Busca correspondência exata (case-insensitive)
        for available in available_genres:
            if genre_lower == available.lower():
                return available

        # Busca correspondência parcial
        for available in available_genres:
            if genre_lower in available.lower() or available.lower() in genre_lower:
                return available

        # Se não encontrar, retorna o gênero original
        return genre

    def set_advanced_processing(self, enabled: bool):
        """Permite habilitar/desabilitar processamento avançado"""
        self.advanced_processing_enabled = enabled

    def set_confidence_threshold(self, threshold: float):
        """Permite ajustar o threshold de confiança"""
        self.confidence_threshold = max(0.1, min(threshold, 0.9))

        # Método para treinar com feedback

    def learn_from_feedback(self, text: str, correct_intent: str, entities: Dict[str, str]):
        """Aprende com feedback do usuário (implementação futura)"""
        # Placeholder para machine learning futuro
        pass

        # Método para processar consultas complexas

    def process_complex_query(self, text: str) -> Dict[str, any]:
        """Processa consultas complexas com múltiplos critérios"""
        result = {
            'primary_intent': self.process(text),
            'multiple_criteria': self._detect_multiple_criteria(text),
            'input_analysis': self._validate_input(text),
            'suggestions': []
        }

        # Se não reconheceu, gera sugestões
        if result['primary_intent'].name == 'unknown':
            result['suggestions'] = self._generate_suggestions(text)

        return result