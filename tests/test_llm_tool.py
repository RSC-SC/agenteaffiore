"""Testes de `src/tools/llm_tool.py`: ordem de fallback, tool-calling e normalização."""
import pytest

from tests.helpers import FakeLLM, FakeResponse, FakeTool, ObjetoComStr


class TestOrdemDeProvedores:
    def test_padrao_e_gemini_groq_openrouter(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.delenv("LLM_PRIMARY_PROVIDER", raising=False)
        assert llm_tool.ordem_provedores() == ["gemini", "groq", "openrouter"]

    @pytest.mark.parametrize(
        ("preferido", "esperado"),
        [
            ("gemini", ["gemini", "groq", "openrouter"]),
            ("groq", ["groq", "gemini", "openrouter"]),
            ("openrouter", ["openrouter", "gemini", "groq"]),
            ("GROQ", ["groq", "gemini", "openrouter"]),
            ("  gemini  ", ["gemini", "groq", "openrouter"]),
        ],
    )
    def test_inverte_conforme_llm_primary_provider(self, monkeypatch, preferido, esperado):
        from src.tools import llm_tool

        monkeypatch.setenv("LLM_PRIMARY_PROVIDER", preferido)
        assert llm_tool.ordem_provedores() == esperado

    def test_valor_invalido_volta_ao_padrao(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "nao-existe")
        assert llm_tool.ordem_provedores() == ["gemini", "groq", "openrouter"]

    def test_ordem_padrao_cobre_todos_os_provedores_declarados(self):
        from src.tools import llm_tool

        assert set(llm_tool.ORDEM_PADRAO) == set(llm_tool.PROVEDORES)

    def test_get_providers_expoe_nome_e_fabrica(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.setenv("LLM_PRIMARY_PROVIDER", "groq")
        pares = llm_tool.get_providers()
        assert [nome for nome, _ in pares] == ["groq", "gemini", "openrouter"]

    def test_todos_os_provedores_falham_no_tipo_certo(self, fake_llm):
        """Trava contra regressão de digitação no nome da exceção."""
        from src.tools import llm_tool

        mapa = {nome: FakeLLM([RuntimeError("erro")]) for nome in ("gemini", "groq", "openrouter")}
        fake_llm(mapa)
        with pytest.raises(llm_tool.TodosProvedoresFalharam):
            llm_tool.chat([])


@pytest.mark.parametrize("nome", ["NenhumProvedorDisponivel", "TodosProvedoresFalharam"])
def test_excecoes_publicas_herdam_de_runtime_error(nome):
    from src.tools import llm_tool

    assert issubclass(getattr(llm_tool, nome), RuntimeError)


class TestNormalizacaoDeConteudo:
    """Regressão do bug em que o Gemini devolve `content` como lista de blocos."""

    @pytest.mark.parametrize(
        ("entrada", "esperado"),
        [
            ("texto simples", "texto simples"),
            ([{"type": "text", "text": "parte 1 "}], "parte 1 "),
            ([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}], "ab"),
            ([{"type": "thinking", "thinking": "..."}, {"type": "text", "text": "visivel"}], "visivel"),
            (["bloco solto"], "bloco solto"),
            ([], ""),
            (123, "123"),
        ],
    )
    def test_texto_normaliza_content(self, entrada, esperado):
        from src.tools import llm_tool

        assert llm_tool._texto(FakeResponse(content=entrada)) == esperado

    def test_texto_trata_objeto_sem_content(self):
        from src.tools import llm_tool

        assert llm_tool._texto(ObjetoComStr()) == "bruto"


class TestFallbackNaInvocacao:
    def test_usa_primeiro_provedor_funcional(self, fake_llm):
        from src.tools import llm_tool

        fake_llm({"gemini": FakeLLM([FakeResponse("ok")]), "groq": None})
        assert llm_tool.chat([]) == ("ok", "gemini")

    def test_avanca_quando_o_cliente_falha_na_chamada(self, fake_llm):
        """O fallback ocorre na invocação, não só na construção do cliente."""
        from src.tools import llm_tool

        fake_llm(
            {
                "gemini": FakeLLM([RuntimeError("RESOURCE_EXHAUSTED")]),
                "groq": FakeLLM([FakeResponse("resposta do fallback")]),
            }
        )
        assert llm_tool.chat([]) == ("resposta do fallback", "groq")

    def test_avanca_por_varios_provedores(self, fake_llm):
        from src.tools import llm_tool

        fake_llm(
            {
                "gemini": FakeLLM([RuntimeError("quota")]),
                "groq": FakeLLM([TimeoutError("timeout")]),
                "openrouter": FakeLLM([FakeResponse("terceiro")]),
            }
        )
        assert llm_tool.chat([]) == ("terceiro", "openrouter")

    def test_erro_nao_vaza_para_o_usuario(self, fake_llm):
        """A exceção que chega ao cliente é genérica; o detalhe fica só no log."""
        from src.tools import llm_tool

        fake_llm(
            {
                "gemini": FakeLLM([RuntimeError("chave sk-secreta em https://api/x")]),
                "groq": FakeLLM([RuntimeError("outro erro interno")]),
                "openrouter": FakeLLM([RuntimeError("mais um")]),
            }
        )
        with pytest.raises(llm_tool.TodosProvedoresFalharam) as exc:
            llm_tool.chat([])
        assert "sk-secreta" not in str(exc.value)
        assert "api/x" not in str(exc.value)
        assert set(exc.value.tentativas) == {"gemini", "groq", "openrouter"}

    def test_lista_de_tentativas_vazia_quando_nada_e_configurado(self, fake_llm):
        from src.tools import llm_tool

        fake_llm({})
        with pytest.raises(llm_tool.NenhumProvedorDisponivel) as exc:
            llm_tool.chat([])
        assert "GOOGLE_API_KEY" in str(exc.value)

    def test_provedor_sem_chave_e_pulado_sem_contar_como_falha(self, fake_llm):
        from src.tools import llm_tool

        fake_llm({"gemini": None, "groq": FakeLLM([FakeResponse("so groq")])})
        assert llm_tool.chat([]) == ("so groq", "groq")


class TestToolCalling:
    def test_executa_a_tool_e_reenvia(self, fake_llm):
        from src.tools import llm_tool

        modelo = FakeLLM(
            [
                FakeResponse(
                    "", tool_calls=[{"name": "obter_links_catalogo", "args": {}, "id": "c1"}]
                ),
                FakeResponse("Aqui estão os links."),
            ]
        )
        fake_llm({"gemini": modelo})

        texto, _ = llm_tool.chat([])
        assert texto == "Aqui estão os links."
        # A segunda chamada inclui um ToolMessage com o conteúdo do catálogo.
        assert any(getattr(m, "type", "") == "tool" for m in modelo.calls[1])

    def test_tool_desconhecida_recebe_resposta_neutra(self, fake_llm):
        """Tool desconhecida não pode deixar a tool_call sem resposta: isso faria o
        modelo devolver texto vazio na segunda chamada."""
        from src.tools import llm_tool

        modelo = FakeLLM(
            [
                FakeResponse("", tool_calls=[{"name": "inexistente", "args": {}, "id": "c1"}]),
                FakeResponse("resposta final"),
            ]
        )
        fake_llm({"gemini": modelo})

        texto, _ = llm_tool.chat([])
        assert texto == "resposta final"
        assert len(modelo.calls) == 2
        # A ToolMessage de erro é anexada ao contexto.
        assert len(modelo.calls[1]) == len(modelo.calls[0]) + 2

    def test_falha_da_tool_nao_injeta_excecao_no_contexto(self, fake_llm):
        """Detalhe técnico da tool não pode entrar no prompt do modelo."""
        from src.tools import llm_tool

        def _explodir(_args):
            raise RuntimeError("detalhe interno da tool: /caminho/secreto")

        modelo = FakeLLM(
            [
                FakeResponse(
                    "", tool_calls=[{"name": "obter_links_catalogo", "args": {}, "id": "c1"}]
                ),
                FakeResponse("desculpe o transtorno"),
            ]
        )
        fake_llm({"gemini": modelo})

        texto, _ = llm_tool.chat([], tools=[FakeTool("obter_links_catalogo", _explodir)])

        assert texto == "desculpe o transtorno"
        conteudos = [str(getattr(m, "content", "")) for m in modelo.calls[1]]
        assert not any("detalhe interno" in c for c in conteudos)
        assert any("Não foi possível consultar" in c for c in conteudos)

    def test_para_ao_limite_de_rodadas(self, fake_llm):
        """Um modelo que re-solicita tool indefinidamente não entra em laço."""
        from src.tools import llm_tool

        modelo = FakeLLM(
            [
                FakeResponse("", tool_calls=[{"name": "obter_links_catalogo", "args": {}, "id": "c"}])
            ]
            * 50
        )
        fake_llm({"gemini": modelo})

        texto, _ = llm_tool.chat(
            [], tools=[FakeTool("obter_links_catalogo", lambda _a: "conteudo")]
        )
        assert isinstance(texto, str)
        assert len(modelo.calls) == llm_tool.MAX_TOOL_ROUNDS


class TestVinculacaoDeTools:
    def test_openrouter_nao_vincula_tools_por_padrao(self, monkeypatch):
        """Modelos :free do OpenRouter não suportam tool-calling de forma confiável."""
        from src.tools import llm_tool

        monkeypatch.delenv("LLM_DISABLE_TOOLS", raising=False)
        assert llm_tool._suporta_tools("openrouter") is False
        assert llm_tool._suporta_tools("gemini") is True
        assert llm_tool._suporta_tools("groq") is True

    def test_llm_disable_tools_desliga_globalmente(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.setenv("LLM_DISABLE_TOOLS", "true")
        assert llm_tool._suporta_tools("gemini") is False
        assert llm_tool._suporta_tools("groq") is False

    def test_bind_tools_aplicado_quando_suportado(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.delenv("LLM_DISABLE_TOOLS", raising=False)
        modelo = FakeLLM()
        llm_tool._vincular(modelo, "gemini", [FakeTool("t", lambda _a: "")])
        assert modelo.tools_vinculadas is not None

    def test_bind_tools_nao_aplicado_no_openrouter(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.delenv("LLM_DISABLE_TOOLS", raising=False)
        modelo = FakeLLM()
        llm_tool._vincular(modelo, "openrouter", [FakeTool("t", lambda _a: "")])
        assert modelo.tools_vinculadas is None


class TestGetLlm:
    def test_devolve_o_primeiro_cliente_construivel(self, fake_llm):
        from src.tools import llm_tool

        fake_llm({"gemini": None, "groq": FakeLLM(nome="groq-fake")})
        assert llm_tool.get_llm([]).nome == "groq-fake"

    def test_sem_nenhum_cliente_levanta(self, fake_llm):
        from src.tools import llm_tool

        fake_llm({})
        with pytest.raises(llm_tool.NenhumProvedorDisponivel):
            llm_tool.get_llm([])


class TestConstrucaoDeCliente:
    def test_sem_chave_retorna_none(self, monkeypatch):
        from src.tools import llm_tool

        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert llm_tool._construir("gemini") is None

    def test_falha_de_import_nao_propaga(self, monkeypatch):
        """Erro ao carregar o pacote do provedor vira None, não exceção."""
        import builtins

        from src.tools import llm_tool

        monkeypatch.setenv("GROQ_API_KEY", "chave-fake")
        real_import = builtins.__import__

        def _import_falho(name, *args, **kwargs):
            if name == "langchain_openai":
                raise ImportError("pacote ausente")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _import_falho)
        assert llm_tool._construir("groq") is None
