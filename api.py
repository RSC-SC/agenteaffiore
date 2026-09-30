# api.py
from fastapi.middleware.cors import CORSMiddleware
from src.tools.observability import get_observer
from src.graph import build_graph
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()


# No topo do api.py, junto aos imports do FastAPI

# Dicionário para guardar o estado de tempo (session_id -> datetime)
historico_sessoes = {}
TEMPO_EXPIRACAO_MINUTOS = 60

app = FastAPI(
    title="API do Agente Conversacional",
    description="Endpoint para integração com n8n e outros sistemas externos.",
    version="1.0.0"
)

# Logo após instanciar: app = FastAPI(...)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite requisições de qualquer navegador/origem durante os testes
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Compila o grafo do LangGraph uma única vez na inicialização
graph = build_graph()

# Memória de sessões em memória (session_id -> lista de mensagens)
session_storage: Dict[str, List[Dict[str, str]]] = {}


# --- Modelos de Entrada e Saída (Pydantic) ---

class ChatRequest(BaseModel):
    message: str = Field(...,
                         description="Mensagem de texto enviada pelo usuário")
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
        raise HTTPException(
            status_code=400, detail="A mensagem não pode estar vazia.")

    # 1. Controle de Tempo e Identificação de Primeira Mensagem
    agora = datetime.now()
    is_primeira_mensagem = False

    if session_id not in historico_sessoes:
        is_primeira_mensagem = True
    else:
        ultima_interacao = historico_sessoes[session_id]
        # Se expirou o tempo limite, tratamos como conversa nova
        if agora - ultima_interacao > timedelta(minutes=TEMPO_EXPIRACAO_MINUTOS):
            is_primeira_mensagem = True
            # Limpa a memória de mensagens antiga do LangGraph
            if session_id in session_storage:
                session_storage[session_id] = []

    # Atualiza o relógio da última interação
    historico_sessoes[session_id] = agora

    # 2. Recupera ou cria o histórico de textos da sessão
    if session_id not in session_storage:
        session_storage[session_id] = []

    history = session_storage[session_id]

    # 3. Injeta contexto dinâmico para a IA se for o início da conversa
    if is_primeira_mensagem:
        print(f"🌟 NOVA CONVERSA INICIADA COM: {session_id}")
        history.append({
            "role": "system",
            "content": "Aviso interno: Esta é a primeira mensagem do usuário nesta interação. Apresente-se ou inicie o fluxo adequadamente."
        })
    else:
        print(f"🔄 CONTINUANDO CONVERSA COM: {session_id}")

    # 4. Adiciona a nova mensagem do usuário
    history.append({"role": "user", "content": user_text})

    # 5. Prepara o estado para o LangGraph (incluindo session_id para a flag DISABLE_LLM)
    state_input = {
        "session_id": session_id,
        "messages": history,
        "is_first_message": is_primeira_mensagem,
        "response": "",
        "error_message": ""
    }

    # 6. Registra observabilidade
    observer = get_observer()
    run_id = observer.start_run()

    try:
        # Executa o grafo
        result_state = graph.invoke(state_input)

        if result_state.get("error_message"):
            observer.finish_run(status="error")
            return ChatResponse(
                session_id=session_id,
                response="",
                error=result_state["error_message"]
            )

        # 7. Atualiza o histórico em memória e extrai a resposta
        session_storage[session_id] = result_state.get("messages", history)

        # Pega a resposta gerada (seja da LLM, mensagem estática ou "" quando silenciado)
        resposta_gerada = result_state.get("response", "")
        if not resposta_gerada and session_storage[session_id] and session_storage[session_id][-1].get("role") == "assistant":
            resposta_gerada = session_storage[session_id][-1].get(
                "content", "")

        observer.finish_run(status="ok")

        return ChatResponse(
            session_id=session_id,
            response=resposta_gerada,
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
    if session_id in historico_sessoes:
        del historico_sessoes[session_id]
    return {"message": f"Sessão {session_id} resetada com sucesso."}
