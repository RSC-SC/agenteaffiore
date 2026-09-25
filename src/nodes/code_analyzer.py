# src/nodes/code_analyzer.py
import logging
import os
import time
from typing import Any, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

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
        logger.warning(f"Falha ao carregar Gemini: {e}")
        return None


def _try_openrouter() -> Optional[BaseChatModel]:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENROUTER_MODEL", "nvidia/nemotron-3-super-120b-a12b:free")
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
        logger.warning(f"Falha ao carregar OpenRouter: {e}")
        return None


def _get_providers():
    providers = [
        ("Gemini", _try_gemini),
        ("OpenRouter", _try_openrouter),
    ]
    primary = (os.getenv("LLM_PRIMARY_PROVIDER") or "gemini").strip().lower()
    if primary == "openrouter":
        providers.reverse()
    elif primary != "gemini":
        logger.warning(
            "LLM_PRIMARY_PROVIDER inválido: '%s'. Usando padrão (gemini).",
            primary,
        )
    return providers        


def responder_chat(state: dict) -> dict:
    """Nó do agente que reage conversando com o usuário usando os provedores com fallback."""
    messages = state.get("messages", [])
    if not messages:
        return {"error_message": "Nenhuma mensagem recebida no estado."}

    # 1. Prompt de sistema orientando a persona
    system_prompt = (
        "Você é um assistente virtual inteligente e prestativo. "
        "Responda ao usuário com clareza, empatia e de forma contextualizada."
    )
    formatted_messages = [SystemMessage(content=system_prompt)]

    # 2. Converte histórico do state para mensagens LangChain
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=content))

    # 3. Itera sobre a lista de provedores tentando inicializar e chamar .invoke()
    last_error = ""
    for name, provider_factory in _get_providers():
        try:
            model_instance = provider_factory()
            if model_instance is None:
                continue

            logger.info("Tentando responder via provedor: %s", name)
            response = model_instance.invoke(formatted_messages)
            ai_reply = response.content if hasattr(response, "content") else str(response)

            return {
                "messages": messages + [{"role": "assistant", "content": ai_reply}],
                "error_message": ""
            }
        except Exception as e:
            logger.warning("Falha ao invocar provedor %s: %s", name, e)
            last_error = f"{name}: {e}"

    return {
        "error_message": f"Nenhum provedor de LLM disponível ou funcional. Último erro: {last_error}"
    }