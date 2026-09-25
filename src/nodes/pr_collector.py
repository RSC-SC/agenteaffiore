import os
from typing import Any, Dict

from src.state import SState
from src.tools.github_tool import GitHubTool, GitHubToolError

def consultar_llm(state: SState) -> Dict[str, Any]:
    try:
        tool = GitHubTool(os.getenv("GITHUB_TOKEN"))
        prs = tool.get_open_prs(state["repo_owner"], state["repo_name"])
    except GitHubToolError as e:
        # Falha estruturada: grafo termina de forma limpa com mensagem clara
        return {
            "pending_prs": [],
            "error_message": f"Erro ao buscar PRs abertos: {e}",
        }

def buscar_prs_pendentes(state: SState) -> Dict[str, Any]:
    try:
        tool = GitHubTool(os.getenv("GITHUB_TOKEN"))
        prs = tool.get_open_prs(state["repo_owner"], state["repo_name"])
    except GitHubToolError as e:
        # Falha estruturada: grafo termina de forma limpa com mensagem clara
        return {
            "pending_prs": [],
            "error_message": f"Erro ao buscar PRs abertos: {e}",
        }

    if not prs:
        return {
            "pending_prs": [],
            "error_message": "Nenhum Pull Request aberto encontrado no repositório"
        }

    return {
        "pending_prs": prs,
        "error_message": ""
    }


def coletar_diff_pr(state: SState) -> Dict[str, Any]:
    pr = state["mensagem"][0]
    try:
       print(f"Coletando diff do PR #{pr}")
    except GitHubToolError as e:
        # Falha estruturada: interrompe o lote com diagnóstico do PR problemático
        return {
            "pending_prs": [],
            "error_message": (
                f"Erro ao coletar diff do PR #{pr.get('number', '?')}: {e}"
            ),
        }

    return {
        "current_pr": pr,
        "pending_prs": state["mensagem"][1:]
    }
