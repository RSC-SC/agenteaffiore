"""Estado compartilhado do grafo LangGraph do Agente Affiore."""
from typing import TypedDict


class ChatState(TypedDict, total=False):
    """Estado de uma conversa.

    ``total=False`` porque o LangGraph exige que todas as chaves sejam
    opcionais: o grafo recebe o estado montado em `api.py`/`main.py` e cada nó
    devolve o dicionário completo. Não há reducer porque não há escrita
    concorrente de chaves (o grafo é linear).

    Atributos:
        messages: Histórico da conversa em ordem cronológica. Entradas de papel
            ``system`` são injetadas pela camada de API e ignoradas na
            formatação para o modelo.
        session_id: Identificador da conversa, usado para memória e para o
            controle de sessões em modo de contingência.
        response: Último texto gerado pela assistente. String vazia quando a
            resposta foi deliberadamente silenciada.
        is_first_message: Marca a primeira interação da sessão.
        error_message: Descrição amigável de falha, pronta para o cliente.
            String vazia quando a execução foi bem-sucedida.
    """

    messages: list[dict[str, str]]
    session_id: str
    response: str
    is_first_message: bool
    error_message: str
