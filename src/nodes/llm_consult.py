# src/nodes/llm_consult.py
import os
from src.state import ChatState

# Dica: Adapte para a LLM que você usa no projeto (ex: langchain_openai, langchain_groq, langchain_google_genai, etc.)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

def consultar_llm(state: ChatState) -> dict:
    """Nó responsável por receber o histórico e gerar a resposta via LLM."""
    messages = state.get("messages", [])
    if not messages:
        return {"error_message": "Nenhuma mensagem fornecida."}

    try:
        # Exemplo com LangChain ChatModel (ou cliente direto da API)
        # from langchain_openai import ChatOpenAI
        # llm = ChatOpenAI(model="gpt-4o-mini")
        
        # Converte as mensagens do estado para o formato do provedor
        # formatted_msgs = [SystemMessage(content="Você é um assistente virtual prestativo.")]
        # for msg in messages:
        #     if msg["role"] == "user":
        #         formatted_msgs.append(HumanMessage(content=msg["content"]))
        #     elif msg["role"] == "assistant":
        #         formatted_msgs.append(AIMessage(content=msg["content"]))
        
        # response = llm.invoke(formatted_msgs)
        # ai_content = response.content

        # Simulação temporária caso ainda não tenha configurado as chaves:
        ai_content = f"Eco inteligente: Recebi sua mensagem '{messages[-1]['content']}' e processei via LLM."

        # Retorna a nova mensagem para ser anexada ao histórico
        return {
            "messages": messages + [{"role": "assistant", "content": ai_content}],
            "error_message": ""
        }
    except Exception as e:
        return {"error_message": f"Erro na chamada da LLM: {str(e)}"}