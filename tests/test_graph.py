"""Testes do grafo LangGraph, do wrapper de instrumentação e do estado."""
import pytest

from tests.helpers import FakeLLM, FakeResponse


class TestMontagemDoGrafo:
    def test_compila_sem_rede_nem_chaves(self, monkeypatch):
        """O build do grafo é usado no boot da API: não pode tocar em rede."""
        for var in ("GOOGLE_API_KEY", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        from src.graph import build_graph

        assert type(build_graph()).__name__ == "CompiledStateGraph"

    def test_grafo_expoe_o_no_responder_chat(self):
        from src.graph import build_graph

        nos = build_graph().get_graph().nodes
        # Ignora os nós internos do LangGraph.
        reais = [n for n in nos if not n.startswith("__")]
        assert reais == ["responder_chat"]

    def test_wrapper_preserva_o_nome_do_no(self):
        """functools.wraps evita que o nó apareça como 'wrapped' no grafo."""
        from src.graph import _instrumentado
        from src.nodes.code_analyzer import responder_chat

        envolvido = _instrumentado("responder_chat", responder_chat)
        assert envolvido.__name__ == "responder_chat"


class TestWrapperDeInstrumentacao:
    def test_mede_latencia_e_publica_eventos(self, monkeypatch, logs_dir):
        from src.graph import _instrumentado
        from src.tools.observability import run_scope

        def _no(state):
            return {**state, "response": "ok", "error_message": ""}

        with run_scope(session_id="s1", logs_dir=logs_dir) as observador:
            observador.start_run(session_id="s1")
            _instrumentado("meu_no", _no)({"messages": []})
            caminhos = observador.finish_run()

        import json
        from pathlib import Path

        eventos = [
            json.loads(linha)
            for linha in Path(caminhos["structured_log"]).read_text(encoding="utf-8").splitlines()
        ]
        nomes = [e["event"] for e in eventos]
        assert "node_start" in nomes and "node_end" in nomes
        fim = next(e for e in eventos if e["event"] == "node_end")
        assert fim["node"] == "meu_no"
        assert fim["status"] == "ok"
        assert fim["duration_ms"] >= 0

    def test_error_do_no_vira_evento_de_erro(self, monkeypatch, logs_dir):
        from src.graph import _instrumentado
        from src.tools.observability import run_scope

        def _no(state):
            return {**state, "response": "", "error_message": "falhou de propósito"}

        with run_scope(session_id="s1", logs_dir=logs_dir) as observador:
            observador.start_run(session_id="s1")
            _instrumentado("meu_no", _no)({"messages": []})
            caminhos = observador.finish_run()

        import json
        from pathlib import Path

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["outcome"] == "failed"
        assert auditoria["nodes_with_errors"] == {"meu_no": 1}

    def test_excecao_e_re_lancada_apos_registrar(self, logs_dir):
        """O wrapper instrumenta e propaga — não engole a exceção."""
        from src.graph import _instrumentado
        from src.tools.observability import run_scope

        def _no(_state):
            raise RuntimeError("falha interna")

        with run_scope(session_id="s1", logs_dir=logs_dir) as observador:
            observador.start_run(session_id="s1")
            with pytest.raises(RuntimeError, match="falha interna"):
                _instrumentado("meu_no", _no)({"messages": []})

    @pytest.mark.parametrize("retorno", [None, [], "texto", 42])
    def test_retorno_nao_dict_e_rejeitado(self, retorno, logs_dir):
        """Regressão: `fn(state) or {}` mascarava retorno None e outros tipos."""
        from src.graph import _instrumentado
        from src.tools.observability import run_scope

        with run_scope(session_id="s1", logs_dir=logs_dir) as observador:
            observador.start_run(session_id="s1")
            with pytest.raises(TypeError, match="deve devolver um dict"):
                _instrumentado("meu_no", lambda _s: retorno)({"messages": []})


class TestExecucaoEndToEnd:
    def test_percorre_o_grafo_com_llm_mockado(self, fake_llm):
        from src.graph import build_graph

        fake_llm({"gemini": FakeLLM([FakeResponse("Olá! Bem-vinda à Affiore.")])})
        grafo = build_graph()

        resultado = grafo.invoke(
            {
                "session_id": "e2e",
                "messages": [{"role": "user", "content": "oi"}],
                "is_first_message": True,
                "response": "",
                "error_message": "",
            }
        )

        assert resultado["response"] == "Olá! Bem-vinda à Affiore."
        assert resultado["error_message"] == ""
        assert resultado["messages"][-1]["role"] == "assistant"

    def test_memoria_cresce_ao_longo_do_dialogo(self, fake_llm):
        """Regressão: sem reducer, o LangGraph substitui a chave inteira. O nó
        precisa devolver a lista completa para o histórico não se perder."""
        from src.graph import build_graph

        fake_llm({"gemini": FakeLLM([FakeResponse("r1"), FakeResponse("r2")])})
        grafo = build_graph()

        estado = {
            "session_id": "e2e2",
            "messages": [{"role": "user", "content": "primeira"}],
            "response": "",
            "error_message": "",
        }
        estado = grafo.invoke(estado)
        estado["messages"].append({"role": "user", "content": "segunda"})
        estado = grafo.invoke(estado)

        # system injetado à parte; no estado: user, assistant, user, assistant
        papeis = [m["role"] for m in estado["messages"]]
        assert papeis == ["user", "assistant", "user", "assistant"]
        assert estado["response"] == "r2"

    def test_erro_de_llm_nao_derruba_o_grafo(self, fake_llm):
        from src.graph import build_graph

        fake_llm({"gemini": None, "groq": None, "openrouter": None})
        grafo = build_graph()

        resultado = grafo.invoke(
            {"session_id": "s", "messages": [{"role": "user", "content": "oi"}]}
        )
        assert resultado["error_message"] == "Atendimento temporariamente indisponível."


class TestEstado:
    def test_chat_state_e_total_false(self):
        """Todas as chaves precisam ser opcionais: o LangGraph exige."""
        assert __import__("src.state", fromlist=["ChatState"]).ChatState.__total__ is False

    def test_declara_todas_as_chaves_usadas_pela_api(self):
        from src.state import ChatState

        esperadas = {"messages", "session_id", "response", "is_first_message", "error_message"}
        assert esperadas <= set(ChatState.__annotations__)

    def test_sem_rede_o_estado_e_um_typed_dict_puro(self):
        from src.state import ChatState

        assert issubclass(ChatState, dict)
