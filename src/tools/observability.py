"""Observabilidade do Agente Affiore — dois sinais correlacionados.

Sinal 1 — LOG ESTRUTURADO (JSONL): ``logs/run_<run_id>.jsonl``
    Um evento JSON por linha, todos com ``run_id`` e ``ts`` (ISO-8601 UTC) e,
    quando aplicável, ``node``, ``session_id`` e ``provider``. Permite
    reconstruir a sequência exata de eventos, erros e latências.

Sinal 2 — REGISTRO DE AUDITORIA (JSON): ``logs/audit_<run_id>.json``
    Consolidação da execução: latência total e por nó (min/média/máx),
    provedores LLM usados, tentativas que falharam (fallback), nós com erro,
    desfecho e o caminho do JSONL do MESMO ``run_id`` — correlação explícita
    entre os dois sinais.

Escopo por execução
-------------------
Cada execução recebe sua própria instância de :class:`RunObserver`, guardada
num :class:`~contextvars.ContextVar`. Isso é essencial para a API HTTP: com um
singleton de processo, requisições simultâneas sobrescreveriam o ``run_id`` um
da outra e misturariam os eventos de clientes diferentes no mesmo arquivo.

Garantias:
- Thread-safe: todas as escritas são protegidas por ``Lock``.
- Best-effort: falha de escrita NUNCA propaga para o fluxo — observabilidade não
  pode derrubar a execução que ela observa.
"""

import json
import logging
import os
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Diretório dos artefatos. Configurável por `LOGS_DIR` para isolar testes
#: e para AllowWrite no container.
LOGS_DIR = os.getenv("LOGS_DIR") or str(Path(__file__).resolve().parents[2] / "logs")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_relpath(path: str) -> str:
    """Caminho relativo ao CWD; absoluto se atravessar unidades (C: vs E:)."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path


class RunObserver:
    """Ciclo de vida dos sinais de observabilidade de UMA execução.

    Use sempre dentro de :func:`run_scope`, que cria, registra no contexto e
    encerra a execução automaticamente::

        with run_scope(session_id="abc") as obs:
            graph.invoke(state)
            obs.mark_status("ok")
    """

    def __init__(self, logs_dir: str = LOGS_DIR) -> None:
        self._logs_dir = str(Path(logs_dir).resolve())
        self._lock = threading.Lock()
        self._active = False
        self._finished = False

        self.run_id: str = ""
        self._jsonl_path: str = ""
        self._audit_path: str = ""
        self._t0_perf: float = 0.0
        self._started_iso: str = ""
        self._session_id: str = ""

        # Agregações consolidadas na auditoria
        self._node_latencies: dict[str, list[float]] = {}
        self._nodes_with_errors: dict[str, int] = {}
        self._llm_providers_used: list[str] = []
        self._llm_failed_attempts: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ #
    # Ciclo de vida
    # ------------------------------------------------------------------ #
    def start_run(self, session_id: str = "") -> str:
        """Abre os dois sinais para uma nova execução e devolve o ``run_id``."""
        with self._lock:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.run_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
            self._jsonl_path = os.path.join(self._logs_dir, f"run_{self.run_id}.jsonl")
            self._audit_path = os.path.join(self._logs_dir, f"audit_{self.run_id}.json")
            self._t0_perf = time.perf_counter()
            self._started_iso = _now_iso()
            self._session_id = session_id
            self._node_latencies = {}
            self._nodes_with_errors = {}
            self._llm_providers_used = []
            self._llm_failed_attempts = []
            self._active = True
            self._finished = False

        self.log_event("run_start", session_id=session_id)
        return self.run_id

    def finish_run(self, status: str = "ok") -> dict[str, str]:
        """Consolida a auditoria, emite ``run_end`` e devolve os caminhos.

        Idempotente: chamadas repetidas são ignoradas, para que o ``run_scope``
        possa servir de rede de segurança sem duplicar a auditoria.
        """
        with self._lock:
            if self._finished:
                return {"structured_log": self._jsonl_path, "audit": self._audit_path}
            self._finished = True

        total_ms = round((time.perf_counter() - self._t0_perf) * 1000, 2)
        self.log_event("run_end", status=status, total_duration_ms=total_ms)

        # Best-effort: falha ao consolidar jamais derruba o fluxo.
        try:
            with self._lock:
                self._active = False
                auditoria: dict[str, Any] = {
                    "run_id": self.run_id,
                    "session_id": self._session_id,
                    "started_at": self._started_iso,
                    "finished_at": _now_iso(),
                    "total_duration_ms": total_ms,
                    "status": "completed" if status == "ok" else "crashed",
                    "outcome": "failed" if self._nodes_with_errors else "succeeded",
                    "nodes_latency": {
                        nome: {
                            "calls": len(valores),
                            "min_ms": round(min(valores), 2),
                            "avg_ms": round(sum(valores) / len(valores), 2),
                            "max_ms": round(max(valores), 2),
                            "total_ms": round(sum(valores), 2),
                        }
                        for nome, valores in sorted(self._node_latencies.items())
                    },
                    "nodes_with_errors": dict(sorted(self._nodes_with_errors.items())),
                    "llm": {
                        "providers_succeeded": self._llm_providers_used,
                        "failed_attempts": self._llm_failed_attempts,
                        "fallback_count": len(self._llm_failed_attempts),
                    },
                    # Correlação explícita entre os dois sinais:
                    "artifacts": {"structured_log": _safe_relpath(self._jsonl_path)},
                }
            os.makedirs(self._logs_dir, exist_ok=True)
            with open(self._audit_path, "w", encoding="utf-8") as arquivo:
                json.dump(auditoria, arquivo, ensure_ascii=False, indent=2)
        except Exception:
            # Não engolido em silêncio: fica registrado sem derrubar a execução.
            logger.warning("Falha ao gravar a auditoria de %s.", self.run_id, exc_info=True)

        return {"structured_log": self._jsonl_path, "audit": self._audit_path}

    # ------------------------------------------------------------------ #
    # Eventos dos nós (instrumentação central em src/graph.py)
    # ------------------------------------------------------------------ #
    def node_started(self, node: str, **fields: Any) -> None:
        self.log_event("node_start", node=node, **fields)

    def node_finished(
        self, node: str, duration_ms: float, status: str, error: str = "", **fields: Any
    ) -> None:
        with self._lock:
            if self._active:
                self._node_latencies.setdefault(node, []).append(duration_ms)
                if status != "ok":
                    self._nodes_with_errors[node] = self._nodes_with_errors.get(node, 0) + 1
        self.log_event(
            "node_end",
            node=node,
            duration_ms=round(duration_ms, 2),
            status=status,
            error=error,
            **fields,
        )

    def log_error(self, source: str, message: str, **fields: Any) -> None:
        """Erro de negócio tratado por um nó (falha estruturada, sem traceback)."""
        self.log_event("error", node=source, message=message, **fields)

    # ------------------------------------------------------------------ #
    # Eventos de domínio
    # ------------------------------------------------------------------ #
    def llm_attempt(self, provider: str, ok: bool, duration_ms: float, error: str = "") -> None:
        """Resultado de uma tentativa de provedor LLM (deixa o fallback visível)."""
        with self._lock:
            if ok:
                if provider not in self._llm_providers_used:
                    self._llm_providers_used.append(provider)
            else:
                self._llm_failed_attempts.append(
                    {
                        "provider": provider,
                        "duration_ms": round(duration_ms, 2),
                        # Truncado e sem detalhe do cliente: pode conter URL de endpoint.
                        "error": error[:200],
                    }
                )
        self.log_event(
            "llm_provider_success" if ok else "llm_provider_result",
            provider=provider,
            duration_ms=round(duration_ms, 2),
            error=error[:200] if error else "",
        )

    # ------------------------------------------------------------------ #
    # Escrita do sinal 1 (JSONL) — thread-safe e best-effort
    # ------------------------------------------------------------------ #
    def log_event(self, event: str, **fields: Any) -> None:
        registro: dict[str, Any] = {
            "ts": _now_iso(),
            "run_id": self.run_id,
            "event": event,
        }
        registro.update({k: v for k, v in fields.items() if v is not None})
        linha = json.dumps(registro, ensure_ascii=False)

        with self._lock:
            if not self._active and event != "run_end":
                return  # execução encerrada ou nunca iniciada: ignora
            try:
                os.makedirs(self._logs_dir, exist_ok=True)
                with open(self._jsonl_path, "a", encoding="utf-8") as arquivo:
                    arquivo.write(linha + "\n")
            except OSError:
                logger.debug("Falha ao gravar evento %s.", event, exc_info=True)


# Instância inerte para chamadas fora de um run_scope: como nunca é iniciada,
# `_active` é False e `log_event` descarta tudo silenciosamente.
_INERTE = RunObserver()
_observador_atual: ContextVar[RunObserver | None] = ContextVar("observador_atual", default=None)


def get_observer() -> RunObserver:
    """Devolve o observador da execução em andamento no contexto atual.

    Fora de um :func:`run_scope`, devolve uma instância inerte (não grava nada).
    """
    return _observador_atual.get() or _INERTE


@contextmanager
def run_scope(session_id: str = "", logs_dir: str | None = None) -> Iterator[RunObserver]:
    """Cria uma execução isolada e garante o encerramento da auditoria.

    Isola requisições concorrentes: cada contexto recebe seu próprio
    ``run_id``, seu JSONL e sua auditoria.

    ``logs_dir`` é resolvido na chamada (e não no import) para que a variável
    de módulo ``LOGS_DIR`` possa ser redefinida — é assim que os testes
    isolam os artefatos em ``tmp_path``.
    """
    observador = RunObserver(logs_dir=logs_dir or LOGS_DIR)
    observador.start_run(session_id=session_id)
    token: Token[RunObserver | None] = _observador_atual.set(observador)
    try:
        yield observador
    finally:
        observador.finish_run(status="ok")
        _observador_atual.reset(token)
