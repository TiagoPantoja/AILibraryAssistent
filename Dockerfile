# Use Python 3.11 slim como base (mais leve e rápido)
FROM python:3.11-slim

# Define o mantenedor
LABEL maintainer="Bookstore AI Team"
LABEL description="IA Híbrida para assistente de livraria virtual"
LABEL version="1.0.0"

# Define variáveis de ambiente
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# Cria usuário não-root para segurança
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Define diretório de trabalho
WORKDIR /app

# Instala dependências do sistema necessárias
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copia e instala dependências Python
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

# Cria diretório para logs
RUN mkdir -p /app/logs

# Define permissões corretas
RUN chown -R appuser:appuser /app
USER appuser

# Expõe a porta da aplicação
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Comando para iniciar a aplicação
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]