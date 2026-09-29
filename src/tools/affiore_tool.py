"""Ferramentas específicas para o atendimento da loja Affiore."""
from langchain_core.tools import tool

# Links oficiais do Google Drive
CATALOGO_PRINCIPAL = "https://drive.google.com/file/d/1jdaGna6hmyQN5ErcPUBjI7rsoFL3UUUU/view?usp=drive_link"
CATALOGO_COMPLEMENTAR = "https://drive.google.com/file/d/13fG-tHpdHBd2WqynAk9PZ2ci8TOcFnv1/view?usp=drive_link"

@tool
def obter_links_catalogo() -> str:
    """Útil quando o cliente solicitar catálogo, cardápio, fotos ou opções de cestas/produtos da Affiore."""
    return (
        "Aqui estão os catálogos completos da Affiore:\n"
        f"• Catálogo Geral & Presentes: {CATALOGO_PRINCIPAL}\n"
        f"• Catálogo Complementar: {CATALOGO_COMPLEMENTAR}"
    )

# Lista consolidada de tools da loja
AFFIORE_TOOLS = [obter_links_catalogo]