import time
from typing import Callable, Optional

from langgraph.graph import END, StateGraph
import src.nodes.code_analyzer

#ALTERAÇÃO: Substituímos o PRReviewState por um ChatState (que você deverá criar
# contendo apenas um array de "messages")[cite: 1].
from src.state import ChatState 
from src.tools.observability import get_observer


def _instrumented(name: str, fn: Callable) -> Callable:
    """Envolve um nó com os sinais de observabilidade."""
    # Mantido o wrapper para registrar o tempo de execução (latência) e erros[cite: 1].
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
        status = "ok"
        obs.node_finished(name, duration_ms, status)
        return result
    return wrapped


def build_graph() -> StateGraph:
    # ALTERAÇÃO: Inicializamos o construtor usando o novo ChatState[cite: 1].
    builder = StateGraph(ChatState)

    # ALTERAÇÃO: Removemos todos os nós antigos de github e adicionamos apenas 
    # o nó que fará a requisição para a LLM[cite: 1].
    builder.add_node("consultar_llm", _instrumented("consultar_llm", src.nodes.code_analyzer.consultar_llm))

    # ALTERAÇÃO: O ponto de entrada agora é direto na consulta à LLM, substituindo 
    # a antiga etapa de "validar_entrada"[cite: 1].
    builder.set_entry_point("consultar_llm")
    
    
    # ALTERAÇÃO: O fluxo vai direto da LLM para o final da execução em cada turno, 
    # removendo as rotas paralelas (fan-out/fan-in) do código original[cite: 1].
    builder.add_edge("consultar_llm", END)

    return builder.compile()
