# src/state.py
from typing import TypedDict, List, Dict, Any

class ChatState(TypedDict):
    """Estado do agente de chat com memória do diálogo."""
    messages: List[Dict[str, str]]
    error_message: str