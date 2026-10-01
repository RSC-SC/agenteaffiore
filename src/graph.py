"""Montagem do grafo LangGraph do Agente Affiore.

O grafo é linear: entrada → `responder_chat` → END. A instrumentação de
observabilidade é aplicada centralmente em `build_graph()`, para que os nós
mantenham-se focados na lógica de negócio.
"""
import functools
import time
from collections.abc import Callable

from langgraph.graph import END, StateGraph

from src.nodes.code_analyzer import responder_chat
from src.state import ChatState
from src.tools.observability import get_observer

NoEstado = ChatState
ResultadoNo = dict


def _instrumentado(nome: str, fn: Callable[[NoEstado], ResultadoNo]) -> Callable[[NoEstado], ResultadoNo]:
    """Envolve um nó com os sinais de observabilidade (latência e erros)."""

    @functools.wraps(fn)
    def envolvido(state: NoEstado) -> ResultadoNo:
        observador = get_observer()
        observador.node_started(nome)
        inicio = time.perf_counter()
        try:
            resultado = fn(state)
        except Exception as exc:
            observador.node_finished(
                nome, (time.perf_counter() - inicio) * 1000, "exception", error=str(exc)
            )
            raise
        if not isinstance(resultado, dict):
            raise TypeError(
                f"O nó {nome!r} deve devolver um dict, recebeu {type(resultado).__name__}."
            )
        latencia_ms = (time.perf_counter() - inicio) * 1000
        erro_no = str(resultado.get("error_message") or "")
        if erro_no:
            observador.log_error(nome, erro_no)
        observador.node_finished(nome, latencia_ms, "error" if erro_no else "ok", error=erro_no)
        return resultado

    return envolvido


def build_graph():
    """Compila o grafo conversacional. Não faz rede nem exige chaves de API."""
    construtor = StateGraph(ChatState)
    construtor.add_node("responder_chat", _instrumentado("responder_chat", responder_chat))
    construtor.set_entry_point("responder_chat")
    construtor.add_edge("responder_chat", END)
    return construtor.compile()
