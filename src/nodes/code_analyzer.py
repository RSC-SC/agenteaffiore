# src/nodes/code_analyzer.py
import logging
import os
import time
from typing import Any, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

# Links dos catálogos no Google Drive fornecidos
CATALOGOS_LINKS = {
    "catalogo_principal": "https://drive.google.com/file/d/1jdaGna6hmyQN5ErcPUBjI7rsoFL3UUUU/view?usp=drive_link",
    "catalogo_complementar": "https://drive.google.com/file/d/13fG-tHpdHBd2WqynAk9PZ2ci8TOcFnv1/view?usp=drive_link"
}

logger = logging.getLogger(__name__)


@tool
def obter_links_catalogo() -> str:
    """Útil para quando o cliente solicitar o catálogo, menu completo em PDF, 
    fotos ou desejar ver as opções completas da Affiore no Google Drive."""
    return (
        "Aqui estão os links oficiais dos nossos catálogos completos da Affiore:\n"
        f"• Catálogo Geral & Presentes: {CATALOGOS_LINKS['catalogo_principal']}\n"
        f"• Catálogo Complementar: {CATALOGOS_LINKS['catalogo_complementar']}\n"
        "Fique à vontade para explorar todos os detalhes com carinho!"
    )


TOOL_MAP = {"obter_links_catalogo": obter_links_catalogo}

SYSTEM_PROMPT_AFFIORE = f"""Você é a assistente virtual da Affiore (Arte em Presentear).
Seu objetivo é atender clientes no WhatsApp de forma acolhedora, afetuosa, elegante e ágil.

Sobre a Affiore:
- Especializada em cestas de café da manhã, boxes de frios, vinhos, kits spa e presentes corporativos.
- Valoriza o afeto em cada detalhe e conta com curadoria de nutricionista (adaptamos para restrições alimentares como opções sem glúten ou lactose sob consulta).
- Todos os produtos possuem taxa de entrega calculada à parte de acordo com a região.

Catálogos Oficiais no Google Drive:
- Catálogo Geral & Presentes: {CATALOGOS_LINKS['catalogo_principal']}
- Catálogo Complementar: {CATALOGOS_LINKS['catalogo_complementar']}

Diretrizes de Atendimento:
1. Seja sempre cordial e atenciosa, utilizando saudações calorosas condizentes com a marca.
2. Quando o cliente pedir para ver o catálogo, fotos ou perguntar sobre os produtos de forma ampla, utilize a ferramenta `obter_links_catalogo` (ou forneça diretamente os links acima) e envie os links do Google Drive.
3. Se o cliente perguntar os preços diretamente, forneça os valores oficiais com clareza:
   - Box Café Seleto: R$ 89,00
   - Mini Box Frios: R$ 89,90
   - Box Afeto e Flores: R$ 209,00
   - Kit Spa: R$ 259,00
   - Affiore Celebrar (Cerveja artesanal IPA): R$ 359,00
   - Box Vinho Affiore (Casillero del Diablo): R$ 389,00
   - Tábuas de Frios: PP (R$ 159,00), P (R$ 199,00), M (R$ 279,00), G (R$ 359,00)
4. Destaque os itens personalizáveis: o cartão de mensagem é cortesia, e itens como canecas com inicial, fotos polaroid, balões e flores podem ser adicionados.
5. Sempre lembre de perguntar para quando seria a entrega e qual a ocasião (aniversário, agradecimento, brinde a dois), para ajudar o cliente a escolher o melhor presente.
"""


def _try_gemini() -> Optional[BaseChatModel]:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key
        )
    except Exception as e:
        logger.debug(f"Falha ao carregar Gemini: {e}")
        return None


def _try_openrouter() -> Optional[BaseChatModel]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENROUTER_MODEL",
                          "nvidia/nemotron-3-super-120b-a12b:free")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/RSC-SC/IADev-MiniProj-Mod2",
                "X-Title": "Agente Revisor de PRs"
            }
        )
    except Exception as e:
        logger.debug(f"Falha ao carregar OpenRouter: {e}")
        return None


def _try_groq() -> Optional[BaseChatModel]:
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
            default_headers={
                "HTTP-Referer": "https://github.com/RSC-SC/IADev-MiniProj-Mod2",
                "X-Title": "Agente Revisor de PRs"
            }
        )
    except Exception as e:
        logger.debug(f"Falha ao carregar groq: {e}")
        return None


def _get_providers():
    providers = [
        ("Gemini", _try_gemini),
        ("GROQ", _try_groq),
        ("OpenRouter", _try_openrouter)
    ]
    primary = (os.getenv("LLM_PRIMARY_PROVIDER") or "gemini").strip().lower()
    if primary == "openrouter":
        providers = [("OpenRouter", _try_openrouter), ("Gemini", _try_gemini), ("GROQ", _try_groq)]
    elif primary == "groq":
        providers = [("GROQ", _try_groq), ("Gemini", _try_gemini), ("OpenRouter", _try_openrouter)]
    elif primary != "gemini":
        logger.debug(
            "LLM_PRIMARY_PROVIDER inválido: '%s'. Usando padrão (gemini).",
            primary,
        )
    return providers


def responder_chat(state: dict) -> dict:
    """Nó do agente que reage conversando com o usuário usando os provedores com fallback silencioso."""
    messages = state.get("messages", [])
    if not messages:
        return {"error_message": "Nenhuma mensagem recebida no estado."}

    # 1. Prompt de sistema orientando a persona da Affiore
    formatted_messages = [SystemMessage(content=SYSTEM_PROMPT_AFFIORE)]

    # 2. Converte histórico do state para mensagens LangChain
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=content))

    # 3. Itera sobre a lista de provedores tentando inicializar e chamar .invoke() silenciosamente
    last_error = ""
    for name, provider_factory in _get_providers():
        try:
            model_instance = provider_factory()
            if model_instance is None:
                continue

            logger.debug("Tentando responder via provedor: %s", name)

            # OpenRouter gratuito não aceita tools (erro 404). Vincula apenas a provedores compatíveis.
            if name in ["Gemini", "GROQ"]:
                active_llm = model_instance.bind_tools([obter_links_catalogo])
            else:
                active_llm = model_instance

            response = active_llm.invoke(formatted_messages)

            # 4. Trata execução de tools se a LLM tiver solicitado
            if hasattr(response, "tool_calls") and response.tool_calls:
                mensagens_com_tools = list(formatted_messages) + [response]

                for call in response.tool_calls:
                    tool_name = call.get("name")
                    tool_args = call.get("args", {})
                    tool_id = call.get("id")

                    tool_fn = TOOL_MAP.get(tool_name)
                    if tool_fn:
                        try:
                            tool_result = tool_fn.invoke(tool_args)
                        except Exception as e:
                            tool_result = f"Erro ao acessar links: {e}"

                        mensagens_com_tools.append(
                            ToolMessage(
                                tool_call_id=tool_id,
                                content=str(tool_result),
                                name=tool_name
                            )
                        )

                final_response = active_llm.invoke(mensagens_com_tools)
                ai_reply = final_response.content if hasattr(final_response, "content") else str(final_response)
            else:
                ai_reply = response.content if hasattr(response, "content") else str(response)

            # Retorno limpo se algum modelo responder com sucesso
            return {
                **state,
                "messages": messages + [{"role": "assistant", "content": ai_reply}],
                "response": ai_reply,
                "error_message": ""
            }
        except Exception as e:
            # Registra como debug sem poluir a saída do usuário
            logger.debug("Falha interna ao tentar provedor %s: %s", name, e)
            last_error = f"{name}: {e}"

    # Se TODOS os provedores falharem, registra o aviso nos logs
    logger.error("Todos os provedores falharam. Último erro: %s", last_error)

    return {
        **state,
        "error_message": f"Nenhum provedor de LLM disponível ou funcional. Último erro: {last_error}"
    }