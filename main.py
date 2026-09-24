# main.py
import os
from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph
from src.tools.observability import get_observer

def main():
    graph = build_graph()
    observer = get_observer()

    run_id = observer.start_run()
    print(f"[obs] Sessão de chat iniciada | run_id={run_id}")
    print("=" * 60)
    print("💬 Agente Chat ativo! Digite sua mensagem (ou 'sair' para encerrar):")
    print("=" * 60)

    state = {
        "messages": [],
        "error_message": ""
    }

    status = "ok"
    try:
        while True:
            user_input = input("\nVocê: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["sair", "exit", "quit"]:
                print("\nEncerrando conversa...")
                break

            # 1. Adiciona a mensagem do usuário ao histórico
            state["messages"].append({"role": "user", "content": user_input})

            # 2. Executa o grafo chamando a LLM de code_analyzer.py
            state = graph.invoke(state)

            # 3. Exibe a réplica gerada
            if state.get("error_message"):
                print(f"[Aviso]: {state['error_message']}")
            else:
                ultima_resposta = state["messages"][-1]["content"]
                print(f"\nAssistente: {ultima_resposta}")

    except (KeyboardInterrupt, EOFError):
        print("\nSessão finalizada pelo usuário.")
    except Exception as e:
        status = "crashed"
        print(f"\n[Erro fatal]: {e}")
    finally:
        paths = observer.finish_run(status=status)
        print("\n" + "=" * 60)
        if isinstance(paths, dict) and "structured_log" in paths:
            print(f"[obs] Registros gravados em: {paths['structured_log']}")
        print("=" * 60)

if __name__ == "__main__":
    main()