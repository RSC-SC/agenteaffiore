from typing import Any, Dict, List, TypedDict


class SState(TypedDict):
    repo_url: str
    repo_owner: str
    repo_name: str
    is_valid: bool
    error_message: str
    pending_prs: List[Dict[str, Any]]
    current_pr: Dict[str, Any]
    current_diff: str
    current_diff_sanitized: str  # diff higienizado anti prompt-injection
    security_report: Dict[str, Any]  # sinais de injeção detectados no diff atual
    current_review: str
    current_metadata_summary: str  # gerado em paralelo à análise do diff
    processed_prs_count: int
    max_prs: int  # limite explícito de iterações do loop (autonomia delimitada)
    dry_run: bool  # limites de autonomia: True = gera revisão, NÃO posta no GitHub
    review_history: List[Dict[str, Any]]
    final_message: str  # mensagem final da execução (sucesso OU falha) — distinta de error_message
    consultar_llm_msg: str  # mensagem de saída do nó consultar_llm (para debug/diagnóstico) 