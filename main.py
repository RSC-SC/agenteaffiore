"""CLI do Agente de Chat Affiore — interação local, sem passar pela API HTTP.

Uso:
    python main.py

Útil para depurar o grafo e ajustar o prompt sem subir o FastAPI. A memória é
apenas da sessão do terminal.
"""
import logging
import os

from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph  # noqa: E402  (load_dotenv precisa vir antes)
from src.tools.observability import run_scope  # noqa: E402

logger = logging.getLogger("affiore.cli")

COMANDOS_SAIDA = {"sair", "exit", "quit", "sai"}
SEPARADOR = "=" * 60


def main() -> int:
    """Executa o loop conversacional. Devolve o código de saída do processo."""
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(levelname)-8s %(message)s",
    )
    grafo = build_graph()

    print(SEPARADOR)
    print("💬 Agente Chat Affiore — digite sua mensagem (ou 'sair' para encerrar)")
    print(SEPARADOR)

    estado = {
        "session_id": "cli",
        "messages": [],
        "is_first_message": True,
        "response": "",
        "error_message": "",
    }

    with run_scope(session_id="cli") as observador:
        while True:
            try:
                entrada = input("\nVocê: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSessão finalizada pelo usuário.")
                break

            if not entrada:
                continue
            if entrada.lower() in COMANDOS_SAIDA:
                print("\nEncerrando conversa.")
                break

            estado["messages"] = [
                *estado.get("messages", []),
                {"role": "user", "content": entrada},
            ]

            try:
                estado = grafo.invoke(estado)
            except Exception:
                observador.finish_run(status="crashed")
                logger.exception("Falha ao invocar o grafo.")
                print("\n[Erro] Falha interna ao processar a mensagem. Encerrando.")
                return 1

            estado["is_first_message"] = False
            erro = estado.get("error_message")
            if erro:
                print(f"\nAssistente: [{erro}]")
            else:
                print(f"\nAssistente: {estado.get('response', '')}")

        caminhos = observador.finish_run(status="ok")

    print(f"\n{SEPARADOR}")
    print(f"[obs] Log estruturado: {caminhos['structured_log']}")
    print(f"[obs] Auditoria      : {caminhos['audit']}")
    print(SEPARADOR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
