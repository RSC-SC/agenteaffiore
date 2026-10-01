"""Fixtures compartilhadas.

Toda a suíte é 100% offline por construção: nenhum teste faz rede e nenhum
provedor de LLM é realmente instanciado. É isso que permite rodar o CI sem
qualquer credencial.
"""
import os

import pytest

os.environ.setdefault("GOOGLE_API_KEY", "")
os.environ.setdefault("GROQ_API_KEY", "")
os.environ.setdefault("OPENROUTER_API_KEY", "")


@pytest.fixture(autouse=True)
def _sem_flags_de_contingencia(monkeypatch):
    """Garante que DISABLE_LLM/LLM_DISABLE_TOOLS não vazem entre testes."""
    monkeypatch.delenv("DISABLE_LLM", raising=False)
    monkeypatch.delenv("STATIC_RESPONSE_MESSAGE", raising=False)
    monkeypatch.delenv("LLM_DISABLE_TOOLS", raising=False)


@pytest.fixture(autouse=True)
def _limpa_sessoes_respondidas():
    """Zera o registro em memória do modo de contingência."""
    from src.nodes import code_analyzer

    code_analyzer._sessoes_respondidas.clear()
    yield
    code_analyzer._sessoes_respondidas.clear()


class FakeResponse:
    """Resposta de LLM controlada por teste."""

    def __init__(self, content="resposta", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """ChatModel falso: devolve uma sequência de respostas ou exceções.

    `bind_tools` devolve `self` para que as asserções sobre `calls` continuem
    válidas depois do vínculo de tools.
    """

    def __init__(self, respostas=None, nome="fake"):
        self.respostas = list(respostas or [FakeResponse("ok")])
        self.calls = []
        self.nome = nome
        self.tools_vinculadas = None

    def bind_tools(self, tools):
        self.tools_vinculadas = list(tools)
        return self

    def invoke(self, mensagens):
        self.calls.append(mensagens)
        if not self.respostas:
            return FakeResponse("resposta padrão")
        proximo = self.respostas.pop(0)
        if isinstance(proximo, Exception):
            raise proximo
        return proximo


@pytest.fixture
def fake_llm(monkeypatch):
    """Registra fakes no lugar dos clientes reais.

    Uso:
        def instalar(mapa):  # {"gemini": FakeLLM([...]), ...}
            monkeypatch.setattr(llm_tool, "_construir", lambda nome: mapa.get(nome))
    """
    from src.tools import llm_tool

    def instalar(mapa):
        monkeypatch.setattr(llm_tool, "_construir", lambda nome: mapa.get(nome))
        return mapa

    return instalar


@pytest.fixture
def logs_dir(tmp_path):
    """Diretório isolado para os artefatos de observabilidade."""
    destino = tmp_path / "logs"
    destino.mkdir()
    return str(destino)


@pytest.fixture
def grafo_conversando(monkeypatch):
    """Grafo cujo único nó devolve uma resposta fixa, sem tocar em LLM.

    O patch é aplicado em `code_analyzer.chat` (e não em `llm_tool.chat`)
    porque `responder_chat` importa o símbolo diretamente: patchar só o módulo
    de origem não teria efeito algum.
    """
    from src.graph import build_graph
    from src.nodes import code_analyzer

    monkeypatch.setattr(
        code_analyzer,
        "chat",
        lambda mensagens, tools=None: (f"resposta({len(mensagens)})", "fake"),
    )
    return build_graph()


@pytest.fixture
def grafo_na_api(monkeypatch):
    """Substitui o grafo da API por um dublê controlado.

    Necessário para testar os caminhos de erro: `api.grafo` é compilado no
    import do módulo, então monkeypatchar um grafo recém-construído não teria
    efeito nenhum sobre os endpoints.
    """
    import api

    class StubGrafo:
        def __init__(self):
            self.estado_recebido = None
            self.erro = None
            self.resposta = "resposta do stub"

        def invoke(self, state):
            self.estado_recebido = state
            if self.erro is not None:
                raise self.erro
            return {**state, "response": self.resposta, "error_message": ""}

    stub = StubGrafo()
    monkeypatch.setattr(api, "grafo", stub)
    return stub


@pytest.fixture
def api_limpa():
    """Zera o estado global da API entre testes."""
    import api

    api.sessoes = api.SessionStore(ttl_minutes=60, max_sessions=100, max_messages=40)
    api.rate_limiter = api.RateLimiter(limite=1000, janela_seg=60)
    yield api
    api.sessoes = api.SessionStore(ttl_minutes=60, max_sessions=100, max_messages=40)


@pytest.fixture
def cliente(api_limpa, monkeypatch, tmp_path):
    """TestClient da API com autenticação desligada e logs isolados."""
    from fastapi.testclient import TestClient

    from src.tools import observability

    monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "")
    monkeypatch.setattr(observability, "LOGS_DIR", str(tmp_path / "logs"))
    with TestClient(api_limpa.app) as c:
        yield c
