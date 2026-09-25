import logging
import re
import time
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class LLMToolError(Exception):
    """Falha estruturada da LLMTool.

    Carrega contexto operacional para tratamento upstream nos nós:
    - operation: qual operação falhou (ex.: 'get_open_prs')
    - status_code: código HTTP quando disponível (None p/ erro local)
    - original_error: mensagem original da exceção
    """

    def __init__(self, operation: str, original_error: str = "",
                 status_code: Optional[int] = None):
        self.operation = operation
        self.status_code = status_code
        self.original_error = original_error
        http_part = f" [HTTP {status_code}]" if status_code else ""
        super().__init__(f"LLMTool falhou em '{operation}'{http_part}: {original_error}")


# Erros HTTP que valem retry (transitórios). Demais (401/403/404/422) são permanentes.
_TRANSIENT_STATUS_CODES = {408, 429, 500, 502, 503, 504}
_REPO_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


class LLMTool:
    """Wrapper resiliente da API LLM (PyGithub).

    Garantias:
    - Validação de entradas antes de qualquer chamada de rede
    - Timeout em todas as requisições
    - Retry limitado com backoff crescente apenas em falhas transitórias
    - Falhas sempre estruturadas como LLMToolError (nunca exceções cruas)
    """

    def __init__(self, token: str, timeout: int = 30,
                 max_retries: int = 3, retry_backoff_seconds: float = 2.0):
        self._validate_token(token)
        self.max_retries = max(1, max_retries)
        self.retry_backoff_seconds = max(0.0, retry_backoff_seconds)
        try:
            self.client = Github(auth=Auth.Token(token), timeout=timeout, per_page=50)
        except Exception as e:
            raise LLMToolError("inicializacao", str(e)) from e

    # ------------------------------------------------------------------ #
    # Validação de entradas
    # ------------------------------------------------------------------ #

    
    # ------------------------------------------------------------------ #
    # Execução com retry limitado e falhas estruturadas
    # ------------------------------------------------------------------ #
    def _execute_with_retry(self, operation: str, func: Callable) -> Any:
        last_error: Optional[LLMToolError] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return func()
            except LLMToolError as e:
                status = getattr(e, "status", None)
                logger.warning(
                    "LLMTool '%s' falhou na tentativa %d/%d (HTTP %s)",
                    operation, attempt, self.max_retries, status,
                )
                last_error = e
                if status not in _TRANSIENT_STATUS_CODES:
                    break  # permanente: não insiste
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)
            except Exception as e:  # erro de rede/local inesperado -> transitório
                logger.warning(
                    "LLMTool '%s' erro inesperado na tentativa %d/%d: %s",
                    operation, attempt, self.max_retries, e,
                )
                wrapped = LLMToolError(status=None, data=str(e))
                last_error = wrapped
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_seconds * attempt)

        raise LLMToolError(
            operation,
            str(getattr(last_error, "data", last_error)),
            getattr(last_error, "status", None),
        )
    
    
