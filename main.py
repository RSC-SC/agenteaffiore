import os
from dotenv import load_dotenv

load_dotenv()
from src.graph import build_graph
from src.tools.observability import get_observer

# ALTERAÇÃO: A função `parse_args` e o uso de `argparse` foram completamente 
# removidos, pois não recebemos mais `repo_url`, `max-prs` ou `dry-run` por 
# linha de comando antes da execução[cite: 2].

def main():
    graph = build_graph()

    # ALTERAÇÃO: O `initial_state` foi simplificado drasticamente. Variáveis como 
    # `pending_prs`, `current_diff`, e `security_report` foram removidas[cite: 2].
    # NOVO: O estado agora foca apenas em gerenciar o histórico de mensagens da conversa.
    state = {
        "messages": []
    }

    # Mantemos o início da observabilidade para rastrear as sessões do chat[cite: 2].
    observer = get_observer()
    run_id = observer.start_run()
    print(f"[obs] run_id={run_id} — chat iniciado")
    print("\n" + "=" * 50)
    print("Chatbot iniciado! (Digite 'sair' para encerrar)")
    print("=" * 50)

    # ALTERAÇÃO: Adicionado um loop 'while True' no lugar da chamada única 
    # de `graph.invoke(initial_state)` com try/except[cite: 2].
    while True:
        try:
            # NOVO: Captura a entrada do usuário de forma interativa.
            user_input = input("\nVocê: ")
            if user_input.lower() in ['sair', 'exit', 'quit']:
                break

            # Adiciona a mensagem do usuário ao estado
            state["messages"].append({"role": "user", "content": user_input})

            # ALTERAÇÃO: Invocamos o grafo a cada nova interação, em vez de 
            # apenas uma vez no início do script[cite: 2].
            state = graph.invoke(state)

            # Extrai e imprime a última mensagem (da LLM) após o processamento
            ultima_mensagem = state["messages"][-1]["content"]
            print(f"Assistente: {ultima_mensagem}")

        except Exception as e:
            # Mantemos a captura de crash inesperado, alertando no console[cite: 2].
            print(f"Erro inesperado: {e}")
            break

    # Mantido o finalizador do observer para salvar logs ao sair do loop[cite: 2].
    observer.finish_run(status="ok")
    print("Chat encerrado.")

if __name__ == "__main__":
    main()