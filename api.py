# api.py
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph
from src.tools.observability import get_observer

app = FastAPI(
    title="API do Agente Conversacional",
    description="Endpoint para integração com n8n e outros sistemas externos.",
    version="1.0.0"
)

# Compila o grafo do LangGraph uma única vez na inicialização
graph = build_graph()

# Memória de sessões em memória (session_id -> lista de mensagens)
# Para produção distribuída, substitua por Redis ou PostgreSQL
session_storage: Dict[str, List[Dict[str, str]]] = {}


# --- Modelos de Entrada e Saída (Pydantic) ---

class ChatRequest(BaseModel):
    message: str = Field(..., description="Mensagem de texto enviada pelo usuário")
    session_id: str = Field(
        default="default_session",
        description="Identificador único da conversa para manter a memória"
    )
    metadata: Optional[Dict[str, str]] = Field(
        default=None,
        description="Metadados adicionais opcionais repassados pelo n8n"
    )

class ChatResponse(BaseModel):
    session_id: str
    response: str
    error: Optional[str] = None


# --- Endpoints ---

@app.get("/health")
def health_check():
    """Endpoint para validação do serviço (health check)."""
    return {"status": "ok", "agent": "active"}


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(payload: ChatRequest):
    """Endpoint principal consumido pelo nó HTTP Request do n8n."""
    session_id = payload.session_id.strip()
    user_text = payload.message.strip()

    if not user_text:
        raise HTTPException(status_code=400, detail="A mensagem não pode estar vazia.")

    # 1. Recupera ou cria o histórico da sessão
    if session_id not in session_storage:
        session_storage[session_id] = []

    history = session_storage[session_id]
    
    # 2. Adiciona a nova mensagem do usuário
    history.append({"role": "user", "content": user_text})

    # 3. Prepara o estado para o LangGraph
    state_input = {
        "messages": history,
        "error_message": ""
    }

    # 4. Registra observabilidade (opcional, mantendo o padrão do projeto)
    observer = get_observer()
    run_id = observer.start_run()

    try:
        # Executa o grafo que invoca o responder_chat do code_analyzer.py
        result_state = graph.invoke(state_input)

        if result_state.get("error_message"):
            observer.finish_run(status="error")
            return ChatResponse(
                session_id=session_id,
                response="",
                error=result_state["error_message"]
            )

        # 5. Atualiza o histórico com a resposta da LLM
        session_storage[session_id] = result_state.get("messages", history)
        ultima_resposta = session_storage[session_id][-1]["content"]

        observer.finish_run(status="ok")

        return ChatResponse(
            session_id=session_id,
            response=ultima_resposta,
            error=None
        )

    except Exception as e:
        observer.finish_run(status="crashed")
        raise HTTPException(
            status_code=500,
            detail=f"Erro interno durante a execução do agente: {str(e)}"
        )


@app.delete("/chat/{session_id}")
def reset_session(session_id: str):
    """Endpoint para limpar a memória de uma sessão específica."""
    if session_id in session_storage:
        del session_storage[session_id]
        return {"message": f"Sessão {session_id} resetada com sucesso."}
    return {"message": "Sessão não encontrada ou já vazia."}