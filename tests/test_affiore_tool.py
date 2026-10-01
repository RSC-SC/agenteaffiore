"""Testes de `src/tools/affiore_tool.py`: catálogo, preços e tools expostas."""
import pytest

from src.tools.affiore_tool import (
    AFFIORE_TOOLS,
    CATALOGO_COMPLEMENTAR,
    CATALOGO_PRINCIPAL,
    CATALOGOS_LINKS,
    PRECO_OFICIAL,
    TOOL_MAP,
    obter_links_catalogo,
    tabela_precos,
)


class TestCatalogo:
    def test_links_sao_urls_do_google_drive(self):
        assert CATALOGO_PRINCIPAL.startswith("https://drive.google.com/file/d/")
        assert CATALOGO_COMPLEMENTAR.startswith("https://drive.google.com/file/d/")

    def test_mapa_de_links_tem_as_duas_entradas(self):
        assert set(CATALOGOS_LINKS) == {"catalogo_principal", "catalogo_complementar"}
        assert CATALOGOS_LINKS["catalogo_principal"] == CATALOGO_PRINCIPAL
        assert CATALOGOS_LINKS["catalogo_complementar"] == CATALOGO_COMPLEMENTAR

    def test_links_sao_distintos(self):
        assert CATALOGO_PRINCIPAL != CATALOGO_COMPLEMENTAR


class TestPrecos:
    def test_tabela_nao_e_vazia(self):
        assert len(PRECO_OFICIAL) > 0

    def test_todo_preco_tem_formato_de_real(self):
        for produto, preco in PRECO_OFICIAL.items():
            assert preco.startswith("R$ "), produto
            assert "," in preco, produto

    def test_texto_da_tabela_inclui_produto_e_preco(self):
        texto = tabela_precos()
        for produto, preco in PRECO_OFICIAL.items():
            assert produto in texto
            assert preco in texto

    def test_texto_da_tabela_usa_um_item_por_linha(self):
        assert len(tabela_precos().splitlines()) == len(PRECO_OFICIAL)


class TestTools:
    def test_tool_exposta_na_lista_padrao(self):
        assert obter_links_catalogo in AFFIORE_TOOLS

    def test_tool_map_esta_indexada_por_nome(self):
        assert set(TOOL_MAP) == {t.name for t in AFFIORE_TOOLS}
        assert TOOL_MAP["obter_links_catalogo"] is obter_links_catalogo

    def test_tool_devolve_os_dois_links(self):
        texto = obter_links_catalogo.invoke({})
        assert CATALOGO_PRINCIPAL in texto
        assert CATALOGO_COMPLEMENTAR in texto

    def test_tool_tem_docstring_para_o_modelo(self):
        """Sem docstring, o LLM não sabe quando chamar a tool."""
        assert obter_links_catalogo.description.strip()
        assert "catálogo" in obter_links_catalogo.description.lower()

    @pytest.mark.parametrize("nome", ["obter_links_catalogo"])
    def test_nome_da_tool_e_estavel(self, nome):
        """O nome é parte do contrato com o modelo — não renomear sem migrar."""
        assert TOOL_MAP[nome].name == nome
