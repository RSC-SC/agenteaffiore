"""Fábrica de modelos LLM com fallback e suporte a tool-calling.

Fonte única de verdade sobre provedores, modelos e ordem de fallback. Nenhum
outro módulo deve instanciar `ChatGoogleGenerativeAI`/`ChatOpenAI`
diretamente — assim evitamos a divergência de defaults que existia entre
`code_analyzer.py` e este módulo antes da consolidação.

Ordem de tentativa: Gemini → Groq → OpenRouter, invertível via
`LLM_PRIMARY_PROVIDER`. O fallback ocorre **na invocação**, não apenas na
construção do cliente: se um provedor responde com erro (quota, 429, timeout),
o próximo da lista é tentado com a mesma conversa.
"""
import logging
import os
import re
import threading
import time
from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.tools import BaseTool

from src.tools.affiore_tool import AFFIORE_TOOLS
from src.tools.observability import get_observer

logger = logging.getLogger(__name__)

# Limite de rodadas de tool-calling por resposta. Evita laço infinito caso um
# modelo re-solicite a mesma tool indefinidamente.
MAX_TOOL_ROUNDS = 3

TEMPERATURA = 0.3

# Modelos :free do OpenRouter em geral não suportam tool-calling de forma
# confiável. Cada entrada declara sua própria capacidade.
PROVEDORES: dict[str, dict[str, Any]] = {
    "gemini": {
        "var_chave": "GOOGLE_API_KEY",
        "var_modelo": "GOOGLE_MODEL",
        "modelo_padrao": "gemini-2.5-flash",
        "suporta_tools": True,
        "fabrica": "langchain_google_genai:ChatGoogleGenerativeAI",
    },
    "groq": {
        "var_chave": "GROQ_API_KEY",
        "var_modelo": "GROQ_MODEL",
        "modelo_padrao": "llama-3.3-70b-versatile",
        "suporta_tools": True,
        "fabrica": "chat_openai:groq",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "openrouter": {
        "var_chave": "OPENROUTER_API_KEY",
        "var_modelo": "OPENROUTER_MODEL",
        "modelo_padrao": "nvidia/nemotron-3-super-120b-a12b:free",
        "suporta_tools": False,
        "fabrica": "chat_openai:openrouter",
        "base_url": "https://openrouter.ai/api/v1",
    },
}

# Ordem padrão de fallback quando LLM_PRIMARY_PROVIDER não é definido.
ORDEM_PADRAO = ("gemini", "groq", "openrouter")

# HTTP que significa "esta chave não volta a funcionar neste processo":
# 401 (credencial inválida/revogada), 402 (sem crédito) e 403 (limite mensal
# excedido — o caso observado com a chave do OpenRouter). Erro transitório é o
# oposto disso: 429 (rate limit), 5xx e timeout merecem nova tentativa na
# próxima mensagem.
STATUS_PERMANENTE = frozenset({401, 402, 403})

_RE_STATUS = re.compile(r"(?:error code|status code)[:\s]+(\d{3})", re.IGNORECASE)

# Provedores cujas chaves falharam de forma permanente. O fallback é tentado a
# cada mensagem, então repetir a chamada para uma chave morta custaria uma
# ida à rede e uma falha por mensagem — sem chance de sucesso. Uma vez
# desativado, o provedor sai da rotação até o processo reiniciar (reprobe).
_desativados: set[str] = set()
_trava_desativados = threading.Lock()


def provedores_desativados() -> list[str]:
    """Provedores fora da rotação por falha permanente de credencial."""
    with _trava_desativados:
        return sorted(_desativados)


def resetar_provedores_desativados() -> None:
    """Reabilita todos os provedores. Usado em teste e ao rodar a CLI."""
    with _trava_desativados:
        _desativados.clear()


def _status_http(exc: Exception) -> int | None:
    """Extrai o status HTTP da exceção, se houver.

    Os SDKs da OpenAI e da LangChain expõem `status_code`; o corpo da resposta
    também traz o número, então há um plano B textual para as exceções
    reempacotadas que perdem o atributo.
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return status
    resposta = getattr(exc, "response", None)
    status = getattr(resposta, "status_code", None)
    if isinstance(status, int):
        return status
    achado = _RE_STATUS.search(str(exc))
    return int(achado.group(1)) if achado else None


def _motivo(exc: Exception) -> str:
    """Classificação curta e legível da falha, para o log do operador."""
    status = _status_http(exc)
    if status in STATUS_PERMANENTE:
        return f"HTTP {status} (credencial/permanente)"
    if status is not None:
        return f"HTTP {status}"
    return type(exc).__name__


def _eh_permanente(exc: Exception) -> bool:
    return _status_http(exc) in STATUS_PERMANENTE


def _desativar(provedor: str, exc: Exception) -> None:
    """Tira o provedor da rotação após falha permanente de credencial."""
    with _trava_desativados:
        if provedor in _desativados:
            return
        _desativados.add(provedor)
    logger.info(
        "Provedor %s desativado neste processo (%s). Reative corrigindo a chave "
        "e reiniciando. Detalhe completo: LOG_LEVEL=DEBUG (API) ou "
        "CLI_LOG_LEVEL=DEBUG (CLI).",
        provedor,
        _motivo(exc),
    )


class NenhumProvedorDisponivel(RuntimeError):
    """Nenhum provedor de LLM pôde ser usado.

    Cobre os dois casos em que não sobrou modelo: nenhuma chave configurada, ou
    todas as chaves configuradas falharam por credencial (401/402/403). Para o
    cliente é a mesma situação — não há modelo utilizável agora — e ele recebe
    a mesma mensagem amigável. A diferença fica para o técnico, que lê
    `desativados` ou a mensagem.
    """

    def __init__(self, mensagem: str, desativados: Sequence[str] = ()) -> None:
        self.desativados = list(desativados)
        super().__init__(mensagem)


class TodosProvedoresFalharam(RuntimeError):
    """Todos os provedores configurados falharam na invocação.

    A mensagem para o usuário final é intencionalmente genérica; o detalhe de
    cada falha fica apenas no log (pode conter URL de endpoint ou trecho de
    credencial).
    """

    def __init__(self, tentativas: Sequence[str]) -> None:
        self.tentativas = list(tentativas)
        super().__init__(
            "Nenhum provedor de LLM respondeu com sucesso "
            f"(tentados: {', '.join(self.tentativas) or 'nenhum'})."
        )


def ordem_provedores() -> list[str]:
    """Devolve a ordem de tentativa conforme `LLM_PRIMARY_PROVIDER`."""
    preferencial = (os.getenv("LLM_PRIMARY_PROVIDER") or "gemini").strip().lower()
    if preferencial in ORDEM_PADRAO:
        return [preferencial] + [p for p in ORDEM_PADRAO if p != preferencial]
    if preferencial:
        logger.warning(
            "LLM_PRIMARY_PROVIDER inválido: %r. Usando ordem padrão (%s).",
            preferencial,
            " → ".join(ORDEM_PADRAO),
        )
    return list(ORDEM_PADRAO)


def _construir(provedor: str) -> BaseChatModel | None:
    """Instancia o client do provedor, ou None se a chave não estiver configurada."""
    spec = PROVEDORES[provedor]
    api_key = os.getenv(spec["var_chave"])
    if not api_key:
        return None

    modelo = os.getenv(spec["var_modelo"], spec["modelo_padrao"])
    try:
        if provedor == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            return ChatGoogleGenerativeAI(
                model=modelo, google_api_key=api_key, temperature=TEMPERATURA
            )

        from langchain_openai import ChatOpenAI

        kwargs: dict[str, Any] = {
            "model": modelo,
            "api_key": api_key,
            "base_url": spec["base_url"],
            "temperature": TEMPERATURA,
        }
        if provedor == "openrouter":
            kwargs["default_headers"] = {
                "HTTP-Referer": "https://github.com/RSC-SC/agenteaffiore",
                "X-Title": "Agente Chat Affiore",
            }
        return ChatOpenAI(**kwargs)
    except Exception:
        logger.warning("Falha ao construir o client de %s.", provedor, exc_info=True)
        return None


def _suporta_tools(provedor: str) -> bool:
    """Se o provedor aceita tool-calling. Pode ser forçado via `LLM_DISABLE_TOOLS`."""
    if (os.getenv("LLM_DISABLE_TOOLS") or "").strip().lower() in {"true", "1", "yes"}:
        return False
    return bool(PROVEDORES[provedor]["suporta_tools"])


def get_providers() -> list[tuple[str, Callable[[], BaseChatModel | None]]]:
    """Devolve [(nome, fábrica)] na ordem de fallback configurada.

    Provedores desativados por falha permanente de credencial ficam de fora.
    Devolve lista vazia se todos caíram — quem decide o erro é `chat()`, que
    conhece também a lista de desativados e monta a mensagem.
    """
    with _trava_desativados:
        fora = set(_desativados)
    return [
        (nome, lambda n=nome: _construir(n))
        for nome in ordem_provedores()
        if nome not in fora
    ]



def get_llm(tools: list[BaseTool] | None = None) -> BaseChatModel:
    """Retorna o primeiro client construível, já com as tools vinculadas quando aplicável.

    Atalho para usos pontuais (execuções pontuais, notebooks). O caminho do
    agente usa `chat()`, que además faz fallback em caso de falha de invocação.
    """
    for nome, fabrica in get_providers():
        client = fabrica()
        if client is None:
            continue
        return _vincular(client, nome, tools if tools is not None else AFFIORE_TOOLS)
    raise NenhumProvedorDisponivel(
        "Configure ao menos uma chave: "
        + ", ".join(spec["var_chave"] for spec in PROVEDORES.values())
    )


def _vincular(
    client: BaseChatModel, provedor: str, tools: list[BaseTool] | None
) -> BaseChatModel:
    if tools and _suporta_tools(provedor):
        return client.bind_tools(tools)
    return client


def _texto(response: Any) -> str:
    """Normaliza o `content` do modelo para texto.

    Modelos Gemini podem devolver `content` como lista de blocos (ContentBlocks)
    em vez de `str`. Sem esta normalização, a resposta chega ao cliente como
    representação de objeto em vez de texto legível.
    """
    conteudo = getattr(response, "content", response)
    if isinstance(conteudo, str):
        return conteudo
    if isinstance(conteudo, list):
        partes = []
        for bloco in conteudo:
            if isinstance(bloco, str):
                partes.append(bloco)
            elif isinstance(bloco, dict) and bloco.get("type") == "text":
                partes.append(bloco.get("text", ""))
        return "".join(partes)
    return str(conteudo)


def _executar_tools(
    llm: BaseChatModel, resposta: AIMessage, tools_por_nome: dict[str, BaseTool]
) -> list[ToolMessage]:
    """Executa as tool-calls solicitadas pelo modelo, isolando falhas individuais.

    Sempre devolve um `ToolMessage` por chamada — inclusive para tools
    desconhecidas ou que falharam. Omitir a resposta deixaria a conversa
    malformada (tool_call sem ToolMessage correspondente) e faria o modelo
    devolver texto vazio.
    """
    mensagens: list[ToolMessage] = []
    for chamada in getattr(resposta, "tool_calls", None) or []:
        nome = chamada.get("name")
        tool = tools_por_nome.get(nome)
        if tool is None:
            logger.warning("Modelo solicitou tool desconhecida: %r.", nome)
            resultado = "Essa funcionalidade não está disponível."
        else:
            try:
                resultado = tool.invoke(chamada.get("args") or {})
            except Exception:
                # Não injeta a exceção no contexto do LLM: devolve instrução neutra
                # e registra o detalhe real apenas no log.
                logger.warning("Falha ao executar a tool %r.", nome, exc_info=True)
                resultado = "Não foi possível consultar este recurso agora."
        mensagens.append(
            ToolMessage(
                tool_call_id=chamada.get("id", ""),
                content=str(resultado),
                name=nome,
            )
        )
    return mensagens


def chat(
    mensagens: Sequence[BaseMessage],
    tools: list[BaseTool] | None = None,
) -> tuple[str, str]:
    """Invoca o LLM percorrendo os provedores até um responder.

    Args:
        mensagens: conversa já formatada em mensagens LangChain.
        tools: tools a vincular. `None` usa as tools padrão da Affiore.

    Returns:
        Tupla ``(texto_da_resposta, nome_do_provedor)``.

    Raises:
        NenhumProvedorDisponivel: nenhuma chave configurada.
        TodosProvedoresFalharam: todos os provedores falharam na invocação.
    """
    tools_efetivas = tools if tools is not None else AFFIORE_TOOLS
    tools_por_nome = {t.name: t for t in tools_efetivas}
    observador = get_observer()
    tentativas: list[str] = []
    algum_configurado = False

    for nome, fabrica in get_providers():
        client = fabrica()
        if client is None:
            continue
        algum_configurado = True
        llm = _vincular(client, nome, tools_efetivas)
        tentativas.append(nome)
        inicio = time.perf_counter()

        try:
            conversa = list(mensagens)
            texto = ""

            for _ in range(MAX_TOOL_ROUNDS):
                resposta = llm.invoke(conversa)
                texto = _texto(resposta)
                tool_calls = getattr(resposta, "tool_calls", None) or []
                if not tool_calls:
                    break
                mensagens_tool = _executar_tools(llm, resposta, tools_por_nome)
                if not mensagens_tool:
                    break
                conversa = [*conversa, resposta, *mensagens_tool]

            observador.llm_attempt(
                nome, ok=True, duration_ms=(time.perf_counter() - inicio) * 1000
            )
            logger.info("Resposta gerada pelo provedor %s.", nome)
            return texto, nome

        except Exception as exc:
            observador.llm_attempt(
                nome, ok=False, duration_ms=(time.perf_counter() - inicio) * 1000, error=str(exc)
            )
            # O fallback é rotina, não incidente: uma linha curta em INFO (oculta
            # no nível padrão) e o detalhe técnico apenas no audit JSONL. O
            # traceback vai para o console só em LOG_LEVEL=DEBUG — no CLI quem
            # lê a tela é o cliente, e ele não deve ver URL de endpoint nem
            # identificador de chave.
            logger.info(
                "Provedor %s falhou (%s); tentando o próximo.",
                nome,
                _motivo(exc),
                exc_info=logger.isEnabledFor(logging.DEBUG),
            )
            if _eh_permanente(exc):
                _desativar(nome, exc)

    desativados = provedores_desativados()
    if algum_configurado and tentativas and all(n in desativados for n in tentativas):
        # Toda chave configurada morreu por credencial. Repetir não adianta: o
        # operador precisa corrigir a chave e reiniciar o processo.
        raise NenhumProvedorDisponivel(
            "Todas as chaves de LLM configuradas falharam por credencial/permissão: "
            f"{', '.join(desativados)}. Reative corrigindo a chave e reiniciando.",
            desativados=desativados,
        )
    if not algum_configurado:
        raise NenhumProvedorDisponivel(
            "Configure ao menos uma chave: "
            + ", ".join(spec["var_chave"] for spec in PROVEDORES.values())
        )
    raise TodosProvedoresFalharam(tentativas)
