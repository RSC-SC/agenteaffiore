"""Nó conversacional do Agente Affiore.

Responsabilidades: montar o prompt da marca, converter o histórico do estado
em mensagens LangChain e devolver a resposta. A escolha de provedor e o
tool-calling ficam em `src/tools/llm_tool.py`; os dados de catálogo e as tools
em `src/tools/affiore_tool.py`.
"""
import logging
import os
from collections import OrderedDict
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.state import ChatState
from src.tools.llm_tool import (
    NenhumProvedorDisponivel,
    TodosProvedoresFalharam,
    chat,
)

logger = logging.getLogger(__name__)

# Capacidade do registro de sessões já atendidas no modo DISABLE_LLM. Um set
# sem limite cresceria indefinidamente em servidor de longa vida.
MAX_SESSOES_REGISTRADAS = 10_000

# Controle em memória das sessões que já receberam a resposta padrão.
_sessoes_respondidas: OrderedDict[str, None] = OrderedDict()


def _registrar_sessao(session_id: str) -> None:
    _sessoes_respondidas[session_id] = None
    _sessoes_respondidas.move_to_end(session_id)
    while len(_sessoes_respondidas) > MAX_SESSOES_REGISTRADAS:
        _sessoes_respondidas.popitem(last=False)


def _system_prompt() -> str:
    """Monta o prompt da Affiore a partir dos dados oficiais de catálogo."""
    from src.tools.affiore_tool import (  # import local evita ciclo com llm_tool
        CATALOGO_COMPLEMENTAR,
        CATALOGO_PRINCIPAL,
        tabela_precos,
    )

    return f"""Você é a assistente virtual da Affiore (Arte em Presentear).
Seu objetivo é atender clientes no WhatsApp de forma acolhedora, afetuosa, elegante e ágil.

Sobre a Affiore:
- Especializada em cesta de café da manhã, boxes de frios, vinhos, kits spa e presentes corporativos.
- Valoriza o afeto em cada detalhe e conta com curadoria de nutricionista (adaptamos para
  restrições alimentares, como opções sem glúten ou lactose, sob consulta).
- Todos os produtos possuem taxa de entrega calculada à parte, de acordo com a região.

Catálogos oficiais no Google Drive:
- Catálogo Geral & Presentes: {CATALOGO_PRINCIPAL}
- Catálogo Complementar: {CATALOGO_COMPLEMENTAR}

Diretrizes de atendimento:
1. Seja sempre cordial e atenciosa, com saudações condizentes com a marca.
2. Quando o cliente pedir catálogo, fotos, cardápio ou a lista completa de produtos, use a
   ferramenta `obter_links_catalogo` e envie os links do Google Drive.
3. Se o cliente perguntar preços, informe os valores oficiais:
{tabela_precos()}
4. Destaque os itens personalizáveis: o cartão de mensagem é cortesia, e itens como canecas
   com inicial, fotos polaroid, balões e flores podem ser adicionados.
5. Pergunte sempre para quando seria a entrega e qual a ocasião (aniversário, agradecimento,
   brinde a dois), para ajudar o cliente a escolher o melhor presente.
6. Não invente produtos, preços, prazos ou condições que não estejam nas informações acima.
   Se não souber, ofereça o catálogo e sugira falar com um atendente humano."""


def _para_mensagens_lc(historico: list[dict[str, str]]) -> list:
    """Converte o histórico do estado em mensagens LangChain."""
    formato = [SystemMessage(content=_system_prompt())]
    for mensagem in historico:
        papel = mensagem.get("role")
        conteudo = mensagem.get("content", "")
        if papel in ("user", "human"):
            formato.append(HumanMessage(content=conteudo))
        elif papel in ("assistant", "ai"):
            formato.append(AIMessage(content=conteudo))
    return formato


def _flag_ativa(nome_var: str) -> bool:
    return (os.getenv(nome_var) or "").strip().lower() in {"true", "1", "yes"}


def _resposta_de_emergencia(state: ChatState, session_id: str) -> dict[str, Any]:
    """Resposta padrão usada quando `DISABLE_LLM` está ativo (modo de contingência)."""
    if session_id in _sessoes_respondidas:
        logger.info("DISABLE_LLM ativo: sessão %s já atendida. Silenciando.", session_id)
        return {**state, "response": "", "error_message": ""}

    _registrar_sessao(session_id)
    mensagem = os.getenv(
        "STATIC_RESPONSE_MESSAGE",
        "Olá! Nosso atendimento automático está temporariamente indisponível. "
        "Em breve um atendente irá falar com você!",
    )
    logger.info("DISABLE_LLM ativo: mensagem padrão para a sessão %s", session_id)
    return {
        **state,
        "messages": [
            *state.get("messages", []),
            {"role": "assistant", "content": mensagem},
        ],
        "response": mensagem,
        "error_message": "",
    }


def responder_chat(state: ChatState) -> dict[str, Any]:
    """Nó do agente: gera uma resposta conversacional a partir do estado.

    Sempre devolve um dicionário completo com ``messages``, ``response`` e
    ``error_message``. Erros de provedor viram mensagem amigável em
    ``error_message``, nunca exceção com detalhe interno.
    """
    historico = state.get("messages") or []
    if not historico:
        return {**state, "response": "", "error_message": "Nenhuma mensagem recebida."}

    session_id = state.get("session_id") or "default_session"

    if _flag_ativa("DISABLE_LLM"):
        return _resposta_de_emergencia(state, session_id)

    try:
        texto, provedor = chat(_para_mensagens_lc(historico))
    except NenhumProvedorDisponivel as exc:
        # Cobre "nenhuma chave configurada" e "toda chave morta por credencial".
        # A mensagem do cliente é a mesma; o técnico lê a exceção, que nomeia os
        # provedores desativados quando é o caso.
        logger.error("Sem provedor de LLM utilizável (sessão %s): %s", session_id, exc)
        return {
            **state,
            "response": "",
            "error_message": "Atendimento temporariamente indisponível.",
        }
    except TodosProvedoresFalharam:
        logger.error("Todos os provedores falharam (sessão %s).", session_id, exc_info=True)
        return {
            **state,
            "response": "",
            "error_message": "Não conseguimos responder agora. Tente novamente em instantes.",
        }

    return {
        **state,
        "messages": [*historico, {"role": "assistant", "content": texto}],
        "response": texto,
        "error_message": "",
    }
