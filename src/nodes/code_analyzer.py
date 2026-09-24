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
        # Refinamento (Issue #16): modelo via env — gemini-2.0-flash foi
        # descontinuado pela Google (HTTP 404 detectado em execução real).
        model = os.getenv("GOOGLE_MODEL", "gemini-3.6-flash")
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
    """Lista de provedores LLM na ordem de tentativa.

    O provedor primário é controlado por LLM_PRIMARY_PROVIDER:
      - ausente/vazio/'gemini' → Gemini primeiro (comportamento padrão);
      - 'openrouter'           → OpenRouter primeiro, Gemini como fallback;
      - outro valor            → padrão (Gemini primeiro), com log de aviso.
    """
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


# NOTA: Mantenha aqui o mesmo cliente LLM que você já usa no arquivo:
# Exemplo se você usa ChatOpenAI / ChatGroq / etc.:
# from langchain_openai import ChatOpenAI
# llm = ...

def responder_chat(state: dict) -> dict:
    """Nó do agente que reage conversando com o usuário usando a LLM existente."""
    llm = _get_providers()
    messages = state.get("messages", [])
    if not messages:
        return {"error_message": "Nenhuma mensagem recebida no estado."}

    # 1. Prompt de sistema para orientar a persona do agente
    system_prompt = (
        "Você é um assistente virtual inteligente e prestativo. "
        "Responda ao usuário com clareza, empatia e de forma contextualizada."
    )
    
    formatted_messages = [SystemMessage(content=system_prompt)]

    # 2. Converte o histórico de mensagens para a LLM
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            formatted_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            formatted_messages.append(AIMessage(content=content))

    try:
        # 3. Invoca a MESMA LLM já instanciada no code_analyzer.py
        # Se você usa `llm.invoke(...)`:
        response = llm.invoke(formatted_messages)
        ai_reply = response.content if hasattr(response, "content") else str(response)

        # 4. Atualiza o estado com a nova réplica da LLM
        return {
            "messages": messages + [{"role": "assistant", "content": ai_reply}],
            "error_message": ""
        }
    except Exception as e:
        return {"error_message": f"Erro ao processar mensagem na LLM: {str(e)}"}