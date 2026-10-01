"""Ferramentas e dados de catálogo do atendimento da loja Affiore.

Fonte única de verdade para os links oficiais do catálogo e para as tools
expostas ao LLM. Cualquier outro módulo que precise falar de catálogo deve
importar daqui, nunca redefinir.
"""
from langchain_core.tools import BaseTool, tool

# --- Catálogos oficiais (Google Drive) ---------------------------------------

CATALOGO_PRINCIPAL = (
    "https://drive.google.com/file/d/1jdaGna6hmyQN5ErcPUBjI7rsoFL3UUUU/view?usp=drive_link"
)
CATALOGO_COMPLEMENTAR = (
    "https://drive.google.com/file/d/13fG-tHpdHBd2WqynAk9PZ2ci8TOcFnv1/view?usp=drive_link"
)

CATALOGOS_LINKS = {
    "catalogo_principal": CATALOGO_PRINCIPAL,
    "catalogo_complementar": CATALOGO_COMPLEMENTAR,
}

# --- Tabela de preços oficial -------------------------------------------------

PRECO_OFICIAL = {
    "Box Café Seleto": "R$ 89,00",
    "Mini Box Frios": "R$ 89,90",
    "Box Afeto e Flores": "R$ 209,00",
    "Kit Spa": "R$ 259,00",
    "Affiore Celebrar (Cerveja artesanal IPA)": "R$ 359,00",
    "Box Vinho Affiore (Casillero del Diablo)": "R$ 389,00",
    "Tábuas de Frios PP": "R$ 159,00",
    "Tábuas de Frios P": "R$ 199,00",
    "Tábuas de Frios M": "R$ 279,00",
    "Tábuas de Frios G": "R$ 359,00",
}


def tabela_precos() -> str:
    """Devolve a tabela de preços oficial em texto legível para o LLM."""
    return "\n".join(f"   - {produto}: {preco}" for produto, preco in PRECO_OFICIAL.items())


def resumo_catalogo() -> str:
    """Devolve o texto de catálogo que a tool `obter_links_catalogo` retorna."""
    return (
        "Aqui estão os catálogos completos da Affiore:\n"
        f"• Catálogo Geral & Presentes: {CATALOGO_PRINCIPAL}\n"
        f"• Catálogo Complementar: {CATALOGO_COMPLEMENTAR}"
    )


# --- Tools expostas ao modelo -------------------------------------------------


@tool
def obter_links_catalogo() -> str:
    """Útil quando o cliente solicitar catálogo, cardápio, fotos ou opções de
    cestas e produtos da Affiore. Use esta ferramenta sempre que o cliente
    quiser ver a lista completa de produtos, preços ou imagens."""
    return resumo_catalogo()


# Lista consolidada de tools da loja
AFFIORE_TOOLS: list[BaseTool] = [obter_links_catalogo]

# Mapa nome -> tool, usado no roteamento do tool-calling
TOOL_MAP: dict[str, BaseTool] = {t.name: t for t in AFFIORE_TOOLS}
