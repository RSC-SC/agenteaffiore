"""API HTTP do Agente de Chat Affiore.

Camada de borda: valida a entrada, mantém a memória de sessão, invoca o grafo
LangGraph e devolve a resposta. Nenhuma lógica de negócio vive aqui — ela está
em `src/nodes/code_analyzer.py` e `src/tools/llm_tool.py`.

Variáveis de ambiente relevantes (ver `.env.example`):
    CORS_ORIGINS: lista separada por vírgula. Vazio = sem CORS (mesma origem).
    API_AUTH_TOKEN: se definido, exige `Authorization: Bearer <token>`.
    RATE_LIMIT_REQUESTS / RATE_LIMIT_WINDOW_SEG: teto por cliente.
    SESSION_TTL_MINUTES / SESSION_MAX: política de memória de sessão.
    LOG_LEVEL: nível do log da aplicação.
"""
import hmac
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from contextlib import contextmanager
from datetime import datetime, timedelta

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.graph import build_graph
from src.tools.observability import run_scope

load_dotenv()

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("affiore.api")

# --- Limites de entrada -------------------------------------------------------

MAX_MENSAGEM_CARACTERES = int(os.getenv("MAX_MENSAGEM_CARACTERES", "4000"))
MAX_ID_SESSAO_CARACTERES = 64

# --- Política de memória de sessão -------------------------------------------

SESSION_TTL_MINUTES = int(os.getenv("SESSION_TTL_MINUTES", "60"))
SESSION_MAX = int(os.getenv("SESSION_MAX", "1000"))
SESSION_MAX_MENSAGENS = int(os.getenv("SESSION_MAX_MENSAGENS", "40"))

# --- Rate limit --------------------------------------------------------------

RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SEG = int(os.getenv("RATE_LIMIT_WINDOW_SEG", "60"))

#: Instrução injetada no início de cada conversa.
_SALUDACAO_SISTEMA = (
    "Aviso interno: Esta é a primeira mensagem do usuário nesta interação. "
    "Apresente-se ou inicie o fluxo adequadamente."
)


def _origens_cors() -> list[str]:
    """Origens permitidas. Vazio = sem CORS, o navegador só acessa a mesma origem."""
    bruto = os.getenv("CORS_ORIGINS", "").strip()
    return [o.strip() for o in bruto.split(",") if o.strip()]


class SessionStore:
    """Memória de sessão com TTL real, teto de capacidade e acesso thread-safe.

    A remoção das entradas expiradas é feita na escrita, e não apenas na leitura:
    um `dict` cuja expiração só é consultada quando a mesma chave é lida novamente
    cresce indefinidamente em servidores de longa vida.
    """

    def __init__(self, ttl_minutes: int, max_sessions: int, max_messages: int) -> None:
        self._ttl = timedelta(minutes=ttl_minutes)
        self._max_sessions = max_sessions
        self._max_messages = max_messages
        self._lock = threading.RLock()
        # session_id -> (última_interação, lista de mensagens)
        self._sessoes: OrderedDict[str, tuple[datetime, list[dict[str, str]]]] = OrderedDict()
        # session_id -> trava, para serializar o ciclo de vida de uma conversa
        self._trancas: OrderedDict[str, threading.Lock] = OrderedDict()

    @contextmanager
    def travar(self, session_id: str):
        """Serializa o processamento de requisições da MESMA sessão.

        Atender uma mensagem é um read-modify-write do histórico (ler, chamar o
        LLM, gravar). Sem esta trava, duas requisições simultâneas na mesma
        conversa leem o mesmo estado e a última gravação descarta a mensagens da
        outra. Sessões diferentes continuam em paralelo.
        """
        with self._lock:
            trava = self._trancas.get(session_id)
            if trava is None:
                trava = threading.Lock()
                self._trancas[session_id] = trava
            self._trancas.move_to_end(session_id)
            while len(self._trancas) > self._max_sessions:
                self._trancas.popitem(last=False)
        trava.acquire()
        try:
            yield
        finally:
            trava.release()

    def _evict_expired(self, agora: datetime) -> None:
        expiradas = [sid for sid, (visto, _) in self._sessoes.items() if agora - visto > self._ttl]
        for sid in expiradas:
            del self._sessoes[sid]

    def _evict_overflow(self) -> None:
        """Descarta as sessões mais antigas até caber no teto."""
        while len(self._sessoes) > self._max_sessions:
            self._sessoes.popitem(last=False)

    def touch(self, session_id: str) -> bool:
        """Registra interação e devolve ``True`` se é a primeira da sessão."""
        agora = datetime.now()
        with self._lock:
            self._evict_expired(agora)
            entrada = self._sessoes.get(session_id)
            if entrada is None:
                self._sessoes[session_id] = (agora, [])
                self._evict_overflow()
                return True
            self._sessoes[session_id] = (agora, entrada[1])
            self._sessoes.move_to_end(session_id)
            return False

    def history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            entrada = self._sessoes.get(session_id)
            return list(entrada[1]) if entrada else []

    def set_history(self, session_id: str, mensagens: list[dict[str, str]]) -> None:
        with self._lock:
            entrada = self._sessoes.get(session_id)
            visto = entrada[0] if entrada else datetime.now()
            self._sessoes[session_id] = (visto, list(mensagens)[-self._max_messages :])
            self._sessoes.move_to_end(session_id)
            self._evict_overflow()

    def reset(self, session_id: str) -> bool:
        with self._lock:
            return self._sessoes.pop(session_id, None) is not None

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "sessoes_ativas": len(self._sessoes),
                "max_sessoes": self._max_sessions,
                "ttl_minutes": int(self._ttl.total_seconds() // 60),
            }


class RateLimiter:
    """Janela deslizante por cliente, em memória do processo.

    Protege o gasto com LLM de abuso por origem única. Não substitui um
    rate limit distribuído quando há múltiplas réplicas.

    As chaves são mantidas em ordem de uso, o que permite descartar as menos
    recentes quando o número de clientes distintos excede `max_chaves`.
    """

    def __init__(self, limite: int, janela_seg: int, max_chaves: int = 10_000) -> None:
        self._limite = limite
        self._janela = janela_seg
        self._max_chaves = max_chaves
        self._lock = threading.Lock()
        self._eventos: OrderedDict[str, deque[float]] = OrderedDict()

    def permitir(self, chave: str) -> bool:
        agora = time.monotonic()
        with self._lock:
            fila = self._eventos.get(chave)
            if fila is None:
                fila = deque()
                self._eventos[chave] = fila
            self._eventos.move_to_end(chave)

            while fila and agora - fila[0] > self._janela:
                fila.popleft()

            # Remove chaves ociosas para que o dicionário não cresça sem teto.
            if len(self._eventos) > self._max_chaves:
                for antiga in [k for k, v in self._eventos.items() if not v]:
                    del self._eventos[antiga]
                while len(self._eventos) > self._max_chaves:
                    self._eventos.popitem(last=False)

            if len(fila) >= self._limite:
                return False
            fila.append(agora)
            return True


# --- Aplicação ---------------------------------------------------------------

app = FastAPI(
    title="Agente de Chat Affiore",
    description="API do assistente virtual da Affiore (Arte em Presentear).",
    version="2.0.0",
)

_origens = _origens_cors()
if _origens:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_origens,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )

#: Grafo compilado uma única vez no boot (sem rede e sem chaves).
grafo = build_graph()

sessoes = SessionStore(SESSION_TTL_MINUTES, SESSION_MAX, SESSION_MAX_MENSAGENS)
rate_limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_SEG)

TOKEN_AUTORIZACAO = os.getenv("API_AUTH_TOKEN", "").strip()
if not TOKEN_AUTORIZACAO:
    logger.warning(
        "API_AUTH_TOKEN não definido: a API está ABERTA. "
        "Defina-o antes de expor o serviço (todo /chat gera custo de LLM)."
    )
if not _origens:
    logger.info("CORS desativado: defina CORS_ORIGINS para habilitar navegador cross-origin.")


# --- Modelos -----------------------------------------------------------------


class ChatRequest(BaseModel):
    """Mensagem do usuário para a assistente."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MENSAGEM_CARACTERES,
        description="Mensagem de texto enviada pelo usuário.",
    )
    session_id: str = Field(
        default="default_session",
        max_length=MAX_ID_SESSAO_CARACTERES,
        description="Identificador da conversa, para manter a memória.",
    )
    metadata: dict[str, str] | None = Field(
        default=None,
        description="Metadados livres repassados pelo cliente (n8n). Não entram no LLM.",
    )


class ChatResponse(BaseModel):
    """Resposta da assistente."""

    session_id: str
    response: str
    provider: str | None = None
    is_first_message: bool = False
    error: str | None = None


# --- Dependências ------------------------------------------------------------


def exigir_autenticacao(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
) -> None:
    """Exige `Authorization: Bearer <token>` quando `API_AUTH_TOKEN` está definido."""
    if not TOKEN_AUTORIZACAO:
        return
    fornecido = ""
    if authorization and authorization.lower().startswith("bearer "):
        fornecido = authorization[7:].strip()
    elif x_api_key:
        fornecido = x_api_key.strip()
    if not secrets_compare(fornecido, TOKEN_AUTORIZACAO):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Credencial ausente ou inválida.",
            headers={"WWW-Authenticate": "Bearer"},
        )


def secrets_compare(a: str, b: str) -> bool:
    """Comparação em tempo constante, para não vazar o token por timing."""
    return bool(a) and hmac.compare_digest(a, b)


# --- Endpoints ---------------------------------------------------------------


@app.get("/health")
def health_check() -> dict:
    """Verificação de disponibilidade e diagnóstico de memória de sessão."""
    return {"status": "ok", "service": "agente-chat-affiore", "sessoes": sessoes.stats()}


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(exigir_autenticacao)])
def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    """Atende uma mensagem do usuário.

    Resposta de negócio com erro sai em HTTP 200 com ``error`` preenchido — é o
    que o n8n espera. Falha inesperada sai em HTTP 500 sem vazar detalhe interno.
    """
    cliente = request.client.host if request.client else "desconhecido"
    if not rate_limiter.permitir(cliente):
        logger.warning("Rate limit excedido para %s.", cliente)
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Muitas requisições. Aguarde alguns instantes.",
            headers={"Retry-After": str(RATE_LIMIT_WINDOW_SEG)},
        )

    session_id = payload.session_id.strip() or "default_session"
    texto = payload.message.strip()

    if not texto:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="A mensagem não pode estar vazia."
        )
    if payload.metadata:
        logger.debug("Metadados recebidos (não repassados ao LLM): %s", payload.metadata.keys())

    # O ciclo inteiro da conversa (ler histórico -> LLM -> gravar) fica sob a
    # trava da sessão: é um read-modify-write e, sem serialização, duas
    # requisições simultâneas na mesma conversa se sobrescrevem.
    with sessoes.travar(session_id):
        primeira = sessoes.touch(session_id)
        historico = sessoes.history(session_id)

        if primeira:
            logger.info("Nova sessão: %s", session_id)
            historico = [
                {"role": "system", "content": _SALUDACAO_SISTEMA},
                *historico,
            ]
        else:
            logger.debug("Sessão em andamento: %s", session_id)

        historico = [*historico, {"role": "user", "content": texto}]

        estado = {
            "session_id": session_id,
            "messages": historico,
            "is_first_message": primeira,
            "response": "",
            "error_message": "",
        }

        with run_scope(session_id=session_id) as observador:
            try:
                resultado = grafo.invoke(estado)
            except Exception:
                observador.finish_run(status="crashed")
                logger.exception("Falha inesperada ao invocar o grafo (sessão %s).", session_id)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Falha interna ao processar a mensagem.",
                ) from None

            mensagens = resultado.get("messages") or historico
            sessoes.set_history(session_id, mensagens)
            erro = resultado.get("error_message") or ""
            if erro:
                observador.finish_run(status="error")

    return ChatResponse(
        session_id=session_id,
        response=resultado.get("response", ""),
        is_first_message=primeira,
        error=erro or None,
    )


@app.delete("/chat/{session_id}", dependencies=[Depends(exigir_autenticacao)])
def reset_session(session_id: str) -> dict:
    """Limpa a memória de uma sessão. Devolve `removed: false` se não existia."""
    alvo = session_id.strip()
    with sessoes.travar(alvo):
        removida = sessoes.reset(alvo)
    logger.info("Reset de sessão %s (removida=%s).", alvo, removida)
    return {"session_id": alvo, "removed": removida}
