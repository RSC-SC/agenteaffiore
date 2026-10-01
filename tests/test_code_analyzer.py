"""Testes de `src/nodes/code_analyzer.py`: formatação de estado e modo de contingência."""
import pytest

from tests.helpers import FakeLLM, FakeResponse


def _estado(mensagens=None, **extra):
    base = {
        "session_id": "s1",
        "messages": mensagens if mensagens is not None else [{"role": "user", "content": "oi"}],
        "response": "",
        "error_message": "",
    }
    return {**base, **extra}


class TestEstadoInvalido:
    def test_sem_mensagens_devolve_erro_amigavel(self):
        from src.nodes.code_analyzer import responder_chat

        resultado = responder_chat(_estado(mensagens=[]))
        assert resultado["error_message"] == "Nenhuma mensagem recebida."
        assert resultado["response"] == ""

    def test_chave_messages_ausente_nao_quebra(self):
        from src.nodes.code_analyzer import responder_chat

        resultado = responder_chat({"session_id": "s1"})
        assert resultado["error_message"] == "Nenhuma mensagem recebida."


class TestFormatacaoDeMensagens:
    def test_system_prompt_e_injetado_primeiro(self, monkeypatch):
        from src.nodes import code_analyzer

        capturado = {}

        def _chat(mensagens, tools=None):
            capturado["mensagens"] = mensagens
            return "ok", "fake"

        monkeypatch.setattr(code_analyzer, "chat", _chat)
        code_analyzer.responder_chat(_estado([{"role": "user", "content": "oi"}]))

        primeira = capturado["mensagens"][0]
        assert primeira.content == code_analyzer._system_prompt()
        assert "Affiore" in primeira.content

    def test_papeis_sao_mapeados_para_langchain(self, monkeypatch):
        from langchain_core.messages import AIMessage, HumanMessage

        from src.nodes import code_analyzer

        capturado = {}

        def _chat(mensagens, tools=None):
            capturado["mensagens"] = mensagens
            return "ok", "fake"

        monkeypatch.setattr(code_analyzer, "chat", _chat)
        historico = [
            {"role": "user", "content": "pergunta"},
            {"role": "assistant", "content": "resposta anterior"},
            {"role": "human", "content": "outra"},
            {"role": "ai", "content": "outra resposta"},
        ]
        code_analyzer.responder_chat(_estado(historico))

        tipos = [type(m).__name__ for m in capturado["mensagens"]]
        # system + user + assistant + human + ai
        assert tipos == ["SystemMessage", "HumanMessage", "AIMessage", "HumanMessage", "AIMessage"]
        assert all(isinstance(m, (HumanMessage, AIMessage)) for m in capturado["mensagens"][1:])

    def test_entrada_de_papel_system_e_ignorada(self, monkeypatch):
        """O aviso interno injetado pela API não deve virar mensagem de sistema dupla."""
        from src.nodes import code_analyzer

        capturado = {}

        def _chat(mensagens, tools=None):
            capturado["mensagens"] = mensagens
            return "ok", "fake"

        monkeypatch.setattr(code_analyzer, "chat", _chat)
        historico = [
            {"role": "system", "content": "aviso interno da API"},
            {"role": "user", "content": "oi"},
        ]
        code_analyzer.responder_chat(_estado(historico))

        tipos = [type(m).__name__ for m in capturado["mensagens"]]
        assert tipos.count("SystemMessage") == 1

    def test_papel_desconhecido_e_descartado(self, monkeypatch):
        from src.nodes import code_analyzer

        capturado = {}

        def _chat(mensagens, tools=None):
            capturado["mensagens"] = mensagens
            return "ok", "fake"

        monkeypatch.setattr(code_analyzer, "chat", _chat)
        code_analyzer.responder_chat(
            _estado([{"role": "tool", "content": "x"}, {"role": "user", "content": "oi"}])
        )
        assert len(capturado["mensagens"]) == 2  # system + user


class TestRespostaComLLM:
    def test_anexa_a_resposta_ao_historico(self, monkeypatch):
        from src.nodes import code_analyzer

        monkeypatch.setattr(code_analyzer, "chat", lambda m, tools=None: ("olá!", "gemini"))
        historico = [{"role": "user", "content": "oi"}]
        resultado = code_analyzer.responder_chat(_estado(historico))

        assert resultado["response"] == "olá!"
        assert resultado["error_message"] == ""
        assert resultado["messages"][-1] == {"role": "assistant", "content": "olá!"}
        assert len(resultado["messages"]) == 2

    def test_nao_sobrescreve_o_historico_original(self, monkeypatch):
        """O dicionário recebido não pode ser mutado no lugar."""
        from src.nodes import code_analyzer

        monkeypatch.setattr(code_analyzer, "chat", lambda m, tools=None: ("r", "fake"))
        historico = [{"role": "user", "content": "oi"}]
        estado = _estado(historico)
        resultado = code_analyzer.responder_chat(estado)

        assert historico == [{"role": "user", "content": "oi"}]
        assert resultado["messages"] is not historico


class TestErrosDeProvedor:
    def test_sem_provedor_configurado(self, monkeypatch):
        from src.nodes import code_analyzer
        from src.tools.llm_tool import NenhumProvedorDisponivel

        def _falhar(m, tools=None):
            raise NenhumProvedorDisponivel("detalhe interno: sem chaves")

        monkeypatch.setattr(code_analyzer, "chat", _falhar)
        resultado = code_analyzer.responder_chat(_estado())
        assert resultado["error_message"] == "Atendimento temporariamente indisponível."
        assert resultado["response"] == ""

    def test_todos_os_provedores_falharam(self, monkeypatch):
        from src.nodes import code_analyzer
        from src.tools.llm_tool import TodosProvedoresFalharam

        def _falhar(m, tools=None):
            raise TodosProvedoresFalharam(["gemini", "groq"])

        monkeypatch.setattr(code_analyzer, "chat", _falhar)
        resultado = code_analyzer.responder_chat(_estado())
        assert "Tente novamente" in resultado["error_message"]

    def test_mensagem_de_erro_nao_contem_detalhe_tecnico(self, monkeypatch):
        from src.nodes import code_analyzer
        from src.tools.llm_tool import TodosProvedoresFalharam

        def _falhar(m, tools=None):
            raise TodosProvedoresFalharam(["gemini"])

        monkeypatch.setattr(code_analyzer, "chat", _falhar)
        resultado = code_analyzer.responder_chat(_estado())
        assert "gemini" not in resultado["error_message"]
        assert "Traceback" not in resultado["error_message"]


class TestModoDeContingencia:
    @pytest.mark.parametrize("valor", ["true", "TRUE", "1", "yes"])
    def test_primeira_mensagem_envia_o_texto_padrao(self, monkeypatch, valor):
        from src.nodes import code_analyzer

        monkeypatch.setenv("DISABLE_LLM", valor)
        resultado = code_analyzer.responder_chat(_estado())
        assert resultado["response"]
        assert resultado["error_message"] == ""
        assert resultado["messages"][-1]["role"] == "assistant"

    def test_mensagem_padrao_pode_ser_personalizada(self, monkeypatch):
        from src.nodes import code_analyzer

        monkeypatch.setenv("DISABLE_LLM", "true")
        monkeypatch.setenv("STATIC_RESPONSE_MESSAGE", "Estamos fora do ar.")
        resultado = code_analyzer.responder_chat(_estado())
        assert resultado["response"] == "Estamos fora do ar."

    def test_segunda_rodada_da_sessao_e_silenciada(self, monkeypatch):
        from src.nodes import code_analyzer

        monkeypatch.setenv("DISABLE_LLM", "true")
        code_analyzer.responder_chat(_estado())
        resultado = code_analyzer.responder_chat(_estado())
        assert resultado["response"] == ""

    def test_sessoes_distintas_recebem_cada_uma_a_sua(self, monkeypatch):
        from src.nodes import code_analyzer

        monkeypatch.setenv("DISABLE_LLM", "true")
        primeiro = code_analyzer.responder_chat(_estado(session_id="a"))
        segundo = code_analyzer.responder_chat(_estado(session_id="b"))
        assert primeiro["response"] and segundo["response"]

    def test_registro_nao_cresce_sem_limite(self, monkeypatch):
        """Guarda contra vazamento de memória do modo de contingência."""
        from src.nodes import code_analyzer

        monkeypatch.setenv("DISABLE_LLM", "true")
        monkeypatch.setattr(code_analyzer, "MAX_SESSOES_REGISTRADAS", 10)
        for i in range(50):
            code_analyzer.responder_chat(_estado(session_id=f"s{i}"))
        assert len(code_analyzer._sessoes_respondidas) <= 10

    def test_flag_desligada_ignora_o_modo_de_contingencia(self, monkeypatch):
        from src.nodes import code_analyzer

        monkeypatch.setenv("DISABLE_LLM", "false")
        monkeypatch.setattr(code_analyzer, "chat", lambda m, tools=None: ("resposta real", "fake"))
        resultado = code_analyzer.responder_chat(_estado())
        assert resultado["response"] == "resposta real"


class TestSystemPrompt:
    def test_inclui_precos_e_links_oficiais(self):
        from src.nodes.code_analyzer import _system_prompt
        from src.tools.affiore_tool import CATALOGO_PRINCIPAL, PRECO_OFICIAL

        prompt = _system_prompt()
        assert CATALOGO_PRINCIPAL in prompt
        for produto, preco in PRECO_OFICIAL.items():
            assert produto in prompt
            assert preco in prompt

    def test_inclui_diretriz_anti_alucinacao(self):
        from src.nodes.code_analyzer import _system_prompt

        assert "Não invente" in _system_prompt()


class TestIntegracaoNoLinhaCompleta:
    def test_fluxo_completo_com_tool_call(self, fake_llm):
        """Integra nó + llm_tool: o catálogo é buscado e a resposta final anexada."""
        from src.nodes import code_analyzer

        modelo = FakeLLM(
            [
                FakeResponse(
                    "", tool_calls=[{"name": "obter_links_catalogo", "args": {}, "id": "c1"}]
                ),
                FakeResponse("Aqui está o catálogo da Affiore."),
            ]
        )
        fake_llm({"gemini": modelo})

        resultado = code_analyzer.responder_chat(_estado())

        assert resultado["response"] == "Aqui está o catálogo da Affiore."
        assert resultado["error_message"] == ""
        # Duas invocações: a que pediu a tool e a que recebeu o resultado.
        assert len(modelo.calls) == 2
