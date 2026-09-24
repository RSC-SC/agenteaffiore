# src/graph.py
import time
from typing import Callable, Optional
from langgraph.graph import END, StateGraph

# REAPROVEITADO: Importando a função do nó diretamente de code_analyzer.py
from src.nodes.code_analyzer import responder_chat
from src.state import ChatState
from src.tools.observability import get_observer


def _instrumented(name: str, fn: Callable) -> Callable:
    """Envolve o nó com os sinais de observabilidade existentes."""
    def wrapped(state: ChatState):
        obs = get_observer()
        obs.node_started(name)
        t0 = time.perf_counter()
        try:
            result = fn(state) or {}
        except Exception as e:
            obs.node_finished(
                name, (time.perf_counter() - t0) * 1000, "exception", error=str(e),
            )
            raise
        duration_ms = (time.perf_counter() - t0) * 1000
        node_error = str(result.get("error_message", "") or "")
        status = "error" if node_error else "ok"
        if node_error:
            obs.log_error(name, node_error)
        obs.node_finished(
            name, duration_ms, status, error=node_error
        )
        return result
    return wrapped


def build_graph() -> StateGraph:
    """Monta o grafo conversacional linear."""
    builder = StateGraph(ChatState)

    # Registra o nó utilizando a LLM do code_analyzer com observabilidade
    builder.add_node("responder_chat", _instrumented("responder_chat", responder_chat))

    # Fluxo direto: Início -> responder_chat -> END
    builder.set_entry_point("responder_chat")
    builder.add_edge("responder_chat", END)

    return builder.compile()