"""Nó de governança: sanitiza o diff externo ANTES de qualquer contato com o LLM.

Posicionamento no grafo:
    coletar_diff_pr → sanitizar_diff ──fan-out──> 
                                              └> resumir_metadados

Garante que nenhum texto não confiável chegue ao modelo sem passar pela
defesa anti prompt-injection (detecção + neutralização + encapsulamento).
"""

import logging
from typing import Any, Dict

from src.state import PRReviewState
from src.tools.observability import get_observer
from src.tools.sanitizer import sanitize_diff

logger = logging.getLogger(__name__)


def sanitizar_diff(state: PRReviewState) -> Dict[str, Any]:
    """Aplica a defesa anti prompt-injection sobre o diff do PR atual.

    O diff é conteúdo externo NÃO CONFIÁVEL. Este nó produz:
      - `current_diff_sanitized`: versão higienizada usada por análise e metadados;
      - `security_report`: relatório estruturado dos sinais detectados
        (consumido pelo `postar_comentario` para transparência no PR).
    """
    diff = state.get("current_diff", "")
    pr_number = (state.get("current_pr", {}) or {}).get("number", "?")

    result = sanitize_diff(diff)

    if result.has_findings:
        logger.warning(
            "Sanitizador (PR #%s): %d sinal(is) de ALTA severidade "
            "(%d linha(s) neutralizada(s)) e %d sinal(is) médio(s).",
            pr_number,
            result.high_signals,
            result.removed_lines,
            result.medium_signals,
        )
    else:
        logger.info("Sanitizador (PR #%s): diff limpo, nenhum padrão suspeito.", pr_number)

    report = {
        "high_signals": result.high_signals,
        "medium_signals": result.medium_signals,
        "removed_lines": result.removed_lines,
        "findings": result.findings,
    }

    # Observabilidade: neutralizações viram evento `security_alert` no log
    # estruturado e entrada na auditoria da execução (correlação por run_id).
    if result.has_findings:
        get_observer().security_alert(
            pr_number if isinstance(pr_number, int) else None,
            high_signals=result.high_signals,
            medium_signals=result.medium_signals,
            removed_lines=result.removed_lines,
        )

    return {
        "current_diff_sanitized": result.sanitized_text,
        "security_report": report,
    }
