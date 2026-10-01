# Imagem de produção do Agente de Chat Affiore.
FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Dependências primeiro: aproveita o cache de camadas do Docker.
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

# Roda sem privilégios. 'affiore' precisa poder escrever em /app/logs.
RUN useradd --create-home --uid 1001 affiore \
    && mkdir -p /app/logs \
    && chown -R affiore:affiore /app
USER affiore

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status==200 else 1)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]