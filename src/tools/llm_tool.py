"""Módulo de inicialização e fábrica de modelos LLM com suporte a Tools."""
import os
import logging
from typing import Optional, List, Any
from dotenv import load_dotenv

from langchain_core.tools import BaseTool

load_dotenv()
logger = logging.getLogger(__name__)

# Importação dos links e tools da loja
try:
    from src.tools.affiore_tool import AFFIORE_TOOLS
except ImportError:
    AFFIORE_TOOLS = []


def _try_gemini():
    """Inicializa o cliente do Google Gemini."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = os.getenv("GOOGLE_MODEL", "gemini-2.5-flash")
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            temperature=0.3,
        )
    except Exception as e:
        logger.warning(f"Falha ao carregar Google Gemini: {e}")
        return None


def _try_groq():
    """Inicializa o cliente da Groq via ChatOpenAI compatível."""
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
            temperature=0.3,
        )
    except Exception as e:
        logger.warning(f"Falha ao carregar Groq: {e}")
        return None


def _try_openrouter():
    """Inicializa o cliente do OpenRouter."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return None
    try:
        from langchain_openai import ChatOpenAI
        model = os.getenv("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
        )
    except Exception as e:
        logger.warning(f"Falha ao carregar OpenRouter: {e}")
        return None


def get_llm(tools: Optional[List[BaseTool]] = None) -> Any:
    """Retorna uma instância funcional de ChatModel com as ferramentas vinculadas."""
    primary = os.getenv("LLM_PRIMARY_PROVIDER", "gemini").lower()
    
    # Ordem de fallback conforme o provedor preferencial
    if primary == "groq":
        providers = [_try_groq, _try_gemini, _try_openrouter]
    elif primary == "openrouter":
        providers = [_try_openrouter, _try_gemini, _try_groq]
    else:  # padrão: gemini
        providers = [_try_gemini, _try_groq, _try_openrouter]

    llm_instance = None
    selected_provider = None
    for provider_func in providers:
        client = provider_func()
        if client is not None:
            llm_instance = client
            selected_provider = provider_func.__name__
            break

    if not llm_instance:
        raise RuntimeError("Nenhum provedor de LLM configurado ou funcional.")

    ferramentas = tools if tools is not None else AFFIORE_TOOLS
    # Modelos :free do OpenRouter normalmente falham em tool_calling.
    # Fazemos bind apenas se não for OpenRouter ou se for provedor com suporte garantido.
    if ferramentas and selected_provider != "_try_openrouter":
        return llm_instance.bind_tools(ferramentas)

    return llm_instance