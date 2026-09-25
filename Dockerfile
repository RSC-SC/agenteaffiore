FROM python:3.11-slim

WORKDIR /app

# Evita criação de arquivos .pyc e força flush imediato de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instala dependências do sistema necessárias para compilação caso precise
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists//*

# Copia e instala dependências Python
COPY requirements.txt requirements-dev.txt* ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir fastapi uvicorn

# Copia todo o código-fonte para o contêiner
COPY . .

# Expõe a porta 8000 do FastAPI
EXPOSE 8000

# Comando para iniciar o servidor Uvicorn
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]