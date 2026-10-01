"""CLI do Agente de Chat Affiore — interação local, sem passar pela API HTTP.

Uso:
    python main.py

Útil para depurar o grafo e ajustar o prompt sem subir o FastAPI. A memória é
apenas da sessão do terminal.
"""
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

from src.graph import build_graph  # noqa: E402  (load_dotenv precisa vir antes)
from src.tools.observability import run_scope  # noqa: E402

logger = logging.getLogger("affiore.cli")

COMANDOS_SAIDA = {"sair", "exit", "quit", "sai"}
SEPARADOR = "=" * 60


def _configurar_saida() -> None:
    """Força UTF-8 na saída padrão.

    O console do Windows vem em cp1252 por padrão, e `print` de emoji ou acento
    estoura `UnicodeEncodeError` — a CLI morre antes de falar com o cliente.
    `errors="replace"` garante que um caractere exótico degrade em vez de
    derrubar a conversa.
    """
    for fluxo in (sys.stdout, sys.stderr):
        reconfigure = getattr(fluxo, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            logger.debug("Não foi possível reconfigurar %s para UTF-8.", fluxo, exc_info=True)


def main() -> int:
    """Executa o loop conversacional. Devolve o código de saída do processo."""
    _configurar_saida()
    # O CLI tem audiência própria: quem lê a tela é o cliente da Affiore, e a
    # troca de provedor é rotina, não incidente. Por isso o padrão é WARNING,
    # separado do LOG_LEVEL da API (lá a tela é do técnico). Para diagnosticar:
    #     CLI_LOG_LEVEL=DEBUG python main.py
    nivel = (os.getenv("CLI_LOG_LEVEL") or "WARNING").upper()
    logging.basicConfig(
        level=getattr(logging, nivel, logging.WARNING),
        format="%(levelname)-8s %(message)s",
        # `force=True` porque `basicConfig` é no-op se o root logger já tiver
        # handler: sem isso, qualquer configuração de log anterior silenciaria
        # o nível do CLI e a tela voltaria a exibir o fallback ao cliente.
        force=True,
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
