"""Testes da CLI (`main.py`) e do silêncio do fallback no console.

O foco é o nível de log: quem lê a tela do CLI é o cliente da Affiore, então o
fallback entre provedores não pode aparecer ali. A troca de provedor é rotina; o
diagnóstico pertence ao log da API (`LOG_LEVEL`) e ao audit JSONL.
"""
import builtins
import logging

import pytest

from tests.helpers import ErroHttp, FakeLLM


@pytest.fixture
def cli(monkeypatch, tmp_path):
    """Executa `main.main()` com terminal, grafo e logs controlados.

    Devolve também o nível do root logger observado durante a execução, porque
    `main()` restaura nada: quem restaura é o `finally` daqui. Medir só depois do
    retorno mediria o nível do pytest, não o do CLI.
    """
    import main as modulo
    from src.nodes import code_analyzer
    from src.tools import observability

    destino = tmp_path / "logs"
    destino.mkdir()
    monkeypatch.setattr(observability, "LOGS_DIR", str(destino))
    monkeypatch.setattr(
        code_analyzer, "chat", lambda _msgs: ("Olá! Como posso ajudar?", "groq")
    )
    raiz = logging.getLogger()
    nivel_anterior = raiz.level
    observado = {}

    def rodar(entradas, nivel=None):
        if nivel is not None:
            monkeypatch.setenv("CLI_LOG_LEVEL", nivel)
        monkeypatch.setattr(builtins, "input", lambda _prompt="": entradas.pop(0))
        try:
            return modulo.main()
        finally:
            observado["nivel"] = raiz.level
            raiz.setLevel(nivel_anterior)

    rodar.nivel_observado = observado
    return rodar


class TestNivelDeLogPadrao:
    def test_padrao_e_warning_mesmo_com_log_level_da_api_em_debug(self, cli, monkeypatch):
        """`LOG_LEVEL` controla a API; o CLI tem audiência própria."""
        monkeypatch.delenv("CLI_LOG_LEVEL", raising=False)
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        assert cli(["sair"]) == 0
        assert cli.nivel_observado["nivel"] == logging.WARNING

    def test_cli_log_level_debug_liberada_o_detalhe(self, cli):
        cli(["olá", "sair"], nivel="DEBUG")
        assert cli.nivel_observado["nivel"] == logging.DEBUG

    def test_nivel_invalido_cai_para_warning(self, cli):
        """`CLI_LOG_LEVEL=barulho` não pode derrubar a CLI nem afrouxar o log."""
        assert cli(["sair"], nivel="barulho") == 0
        assert cli.nivel_observado["nivel"] == logging.WARNING

    def test_o_nivel_do_cli_sobrescreve_configuracao_anterior(self, cli):
        """Regressão do ruído: `basicConfig` é no-op se já houver handler.

        Sem `force=True`, qualquer biblioteca que configurasse logging antes
        silenciaria o nível do CLI e a tela voltaria a mostrar o traceback do
        fallback ao cliente.
        """
        logging.getLogger().addHandler(logging.NullHandler())
        try:
            cli(["sair"], nivel="DEBUG")
        finally:
            logging.getLogger().handlers.clear()
        assert cli.nivel_observado["nivel"] == logging.DEBUG


class TestFallbackSilencioso:
    """Regressão do ruído visto em produção: traceback na tela do cliente."""

    @pytest.fixture
    def provedores_quebrados(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.setattr(llm_tool, "ordem_provedores", lambda: ["groq", "gemini"])
        monkeypatch.setattr(
            llm_tool,
            "_construir",
            lambda nome: {
                "groq": FakeLLM([ErroHttp(403)], nome="groq"),
                "gemini": FakeLLM(nome="gem"),
            }.get(nome),
        )
        return llm_tool

    def test_nada_acima_de_warning_no_nivel_padrao(self, provedores_quebrados, caplog):
        with caplog.at_level(logging.WARNING):
            texto, provedor = provedores_quebrados.chat([])
        assert (texto, provedor) == ("ok", "gemini")
        visiveis = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not visiveis, [r.getMessage() for r in visiveis]
        assert all(r.exc_info is None for r in caplog.records)

    def test_a_falha_continua_registrada_para_o_tecnico(
        self, provedores_quebrados, caplog
    ):
        with caplog.at_level(logging.INFO):
            provedores_quebrados.chat([])
        mensagens = [r.getMessage() for r in caplog.records]
        assert any("403" in m for m in mensagens)

    def test_o_erro_nao_entra_no_texto_que_vai_ao_provedor_seguinte(
        self, provedores_quebrados
    ):
        from langchain_core.messages import HumanMessage

        groq = provedores_quebrados._construir("groq")
        gemini = provedores_quebrados._construir("gemini")
        provedores_quebrados._construir = lambda nome: {"groq": groq, "gemini": gemini}[
            nome
        ]
        provedores_quebrados.chat([HumanMessage("olá")])
        enviado = gemini.calls[0]
        assert all("403" not in str(m.content) for m in enviado)


class TestLoopConversacional:
    def test_encerra_e_devolve_zero(self, cli, capsys):
        assert cli(["olá", "sair"]) == 0
        assert "Olá! Como posso ajudar?" in capsys.readouterr().out

    def test_aponta_o_caminho_dos_artefatos(self, cli, capsys):
        cli(["olá", "sair"])
        saida = capsys.readouterr().out
        assert "Log estruturado" in saida
        assert "Auditoria" in saida

    @pytest.mark.parametrize("comando", ["sair", "SAIR", "exit", "quit", " sai "])
    def test_comandos_de_encerramento(self, cli, comando):
        assert cli([comando]) == 0

    def test_entrada_vazia_nao_encerra(self, cli):
        assert cli(["", "   ", "olá", "sair"]) == 0

    def test_erro_interno_devolve_um_e_nao_vaza_detalhe(self, cli, capsys, monkeypatch):
        class GrafoQuebrado:
            def invoke(self, _estado):
                raise RuntimeError("detalhe interno do fornecedor: sk-secreta")

        monkeypatch.setattr("main.build_graph", lambda: GrafoQuebrado())
        assert cli(["olá"]) == 1
        assert "sk-secreta" not in capsys.readouterr().out
