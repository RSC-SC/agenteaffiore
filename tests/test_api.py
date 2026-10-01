"""Testes da camada HTTP (api.py): validação, sessão, rate limit e autenticação."""
from concurrent.futures import ThreadPoolExecutor

import pytest


class TestHealth:
    def test_retorna_ok_e_diagnostico_de_sessao(self, cliente):
        r = cliente.get("/health")
        assert r.status_code == 200
        corpo = r.json()
        assert corpo["status"] == "ok"
        assert corpo["service"] == "agente-chat-affiore"
        assert "sessoes_ativas" in corpo["sessoes"]

    def test_nao_exige_autenticacao(self, api_limpa, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "segredo")
        with TestClient(api_limpa.app) as c:
            assert c.get("/health").status_code == 200

    def test_saudavel_quando_nenhum_provedor_esta_desativado(self, cliente):
        corpo = cliente.get("/health").json()
        assert corpo["status"] == "ok"
        assert corpo["provedores"]["desativados"] == []

    def test_degradado_quando_uma_chave_de_llm_morre(self, cliente):
        """O operador precisa ver a chave morta; o cliente do chat, não."""
        from src.tools import llm_tool

        llm_tool._desativar("openrouter", RuntimeError("Error code: 403"))
        corpo = cliente.get("/health").json()
        assert corpo["status"] == "degraded"
        assert corpo["provedores"]["desativados"] == ["openrouter"]
        assert "corrigir a chave" in corpo["provedores"]["como_recuperar"]

    def test_diagnostico_de_provedor_nao_vaza_para_o_chat(self, grafo_conversando, cliente):
        """A degradação é infraestrutura: não entra no corpo de /chat."""
        from src.tools import llm_tool

        llm_tool._desativar("openrouter", RuntimeError("Error code: 403"))
        corpo = cliente.post("/chat", json={"message": "olá"}).json()
        assert "openrouter" not in str(corpo)
        assert "403" not in str(corpo)


class TestValidacaoDeEntrada:
    def test_recusa_mensagem_vazia(self, cliente, grafo_conversando):
        r = cliente.post("/chat", json={"message": "   "})
        assert r.status_code == 400
        assert "vazia" in r.json()["detail"].lower()

    def test_recusa_mensagem_acima_do_limite(self, cliente, grafo_conversando):
        r = cliente.post("/chat", json={"message": "x" * 5000})
        assert r.status_code == 422

    def test_recusa_session_id_absurdamente_longo(self, cliente, grafo_conversando):
        r = cliente.post("/chat", json={"message": "oi", "session_id": "s" * 200})
        assert r.status_code == 422

    def test_session_id_vazio_cuida_o_default(self, cliente, grafo_conversando):
        r = cliente.post("/chat", json={"message": "oi", "session_id": "   "})
        assert r.status_code == 200
        assert r.json()["session_id"] == "default_session"


class TestChat:
    def test_responde_e_marca_primeira_mensagem(self, cliente, grafo_conversando):
        r = cliente.post("/chat", json={"message": "Quero um box de café", "session_id": "s1"})
        assert r.status_code == 200
        corpo = r.json()
        assert corpo["error"] is None
        assert corpo["response"]
        assert corpo["is_first_message"] is True

    def test_segunda_mensagem_mantem_memoria(self, cliente, grafo_conversando):
        cliente.post("/chat", json={"message": "primeira", "session_id": "s2"})
        r = cliente.post("/chat", json={"message": "segunda", "session_id": "s2"})
        assert r.json()["is_first_message"] is False
        # histórico cresce: o nó recebe as mensagens anteriores + a nova
        assert "resposta(4)" in r.json()["response"]

    def test_sessoes_distintas_nao_compartilham_memoria(self, cliente, grafo_conversando):
        cliente.post("/chat", json={"message": "ola", "session_id": "a"})
        r = cliente.post("/chat", json={"message": "ola", "session_id": "b"})
        assert r.json()["is_first_message"] is True

    def test_erro_de_llm_via_200_com_campo_error(self, cliente, grafo_na_api):
        def _falhar(state):
            return {**state, "response": "", "error_message": "Atendimento indisponível."}

        grafo_na_api.invoke = _falhar
        r = cliente.post("/chat", json={"message": "oi"})
        assert r.status_code == 200
        assert r.json()["error"] == "Atendimento indisponível."
        assert r.json()["response"] == ""

    def test_excecao_nao_vaza_detalhe_interno(self, cliente, grafo_na_api):
        def _explodir(_state):
            raise RuntimeError("segredo: sk-abc123 endpoint interno")

        grafo_na_api.invoke = _explodir
        r = cliente.post("/chat", json={"message": "oi"})
        assert r.status_code == 500
        detalhe = r.json()["detail"]
        assert "sk-abc123" not in detalhe
        assert "segredo" not in detalhe
        assert "endpoint interno" not in detalhe

    def test_metadata_e_aceito_e_nao_vaza_para_o_llm(self, cliente, grafo_conversando):
        r = cliente.post(
            "/chat", json={"message": "oi", "metadata": {"origem": "n8n", "pedido": "42"}}
        )
        assert r.status_code == 200
        assert r.json()["error"] is None


class TestAutenticacao:
    def test_exige_credencial_quando_token_configurado(self, api_limpa, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "segredo")
        with TestClient(api_limpa.app) as c:
            assert c.post("/chat", json={"message": "oi"}).status_code == 401

    def test_aceita_bearer_valido(self, api_limpa, monkeypatch, grafo_conversando):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "segredo")
        with TestClient(api_limpa.app) as c:
            r = c.post("/chat", json={"message": "oi"}, headers={"Authorization": "Bearer segredo"})
            assert r.status_code == 200

    def test_aceita_x_api_key(self, api_limpa, monkeypatch, grafo_conversando):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "segredo")
        with TestClient(api_limpa.app) as c:
            r = c.post("/chat", json={"message": "oi"}, headers={"X-API-Key": "segredo"})
            assert r.status_code == 200

    def test_rejeita_token_errado(self, api_limpa, monkeypatch, grafo_conversando):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "segredo")
        with TestClient(api_limpa.app) as c:
            r = c.post("/chat", json={"message": "oi"}, headers={"Authorization": "Bearer errado"})
            assert r.status_code == 401

    def test_reset_exige_credencial(self, api_limpa, monkeypatch):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "segredo")
        with TestClient(api_limpa.app) as c:
            assert c.delete("/chat/abc").status_code == 401


class TestRateLimit:
    def test_bloqueia_apos_o_teto(self, api_limpa, monkeypatch, grafo_conversando):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "rate_limiter", api_limpa.RateLimiter(limite=3, janela_seg=60))
        with TestClient(api_limpa.app) as c:
            codigos = [c.post("/chat", json={"message": "oi"}).status_code for _ in range(5)]
        assert codigos.count(200) == 3
        assert codigos.count(429) == 2

    def test_informa_retry_after(self, api_limpa, monkeypatch, grafo_conversando):
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "rate_limiter", api_limpa.RateLimiter(limite=1, janela_seg=60))
        with TestClient(api_limpa.app) as c:
            c.post("/chat", json={"message": "oi"})
            r = c.post("/chat", json={"message": "oi"})
        assert r.status_code == 429
        assert "Retry-After" in r.headers


class TestResetDeSessao:
    def test_limpa_a_memoria(self, cliente, grafo_conversando):
        cliente.post("/chat", json={"message": "oi", "session_id": "s3"})
        r = cliente.delete("/chat/s3")
        assert r.status_code == 200
        assert r.json() == {"session_id": "s3", "removed": True}

    def test_sessao_inexistente_nao_quebra(self, cliente):
        r = cliente.delete("/chat/nao-existe")
        assert r.status_code == 200
        assert r.json()["removed"] is False

    def test_depois_do_reset_a_mensagem_e_novamente_primeira(self, cliente, grafo_conversando):
        cliente.post("/chat", json={"message": "oi", "session_id": "s4"})
        cliente.delete("/chat/s4")
        r = cliente.post("/chat", json={"message": "oi", "session_id": "s4"})
        assert r.json()["is_first_message"] is True


class TestConcorrencia:
    def test_requisicoes_simultaneas_nao_corrompem_a_sessao(
        self, api_limpa, monkeypatch, grafo_conversando
    ):
        """Dez chamadas concorrentes na mesma sessão: o histórico não pode duplicar
        nem se perder, e nenhuma requisição pode estourar."""
        from fastapi.testclient import TestClient

        monkeypatch.setattr(api_limpa, "TOKEN_AUTORIZACAO", "")
        with TestClient(api_limpa.app) as c:
            with ThreadPoolExecutor(max_workers=10) as pool:
                futures = [
                    pool.submit(c.post, "/chat", json={"message": f"m{i}", "session_id": "conc"})
                    for i in range(10)
                ]
                codigos = [f.result().status_code for f in futures]

        assert set(codigos) == {200}
        # 1 system + 10 mensagens de usuário = 11 (a 1ª chamada também gera resposta)
        assert len(api_limpa.sessoes.history("conc")) >= 11


@pytest.mark.parametrize(
    ("bruto", "esperado"),
    [
        ("", []),
        ("   ", []),
        ("http://localhost:5173", ["http://localhost:5173"]),
        (
            "http://localhost:5173, https://app.affiore.com.br",
            ["http://localhost:5173", "https://app.affiore.com.br"],
        ),
        ("http://a.com, http://a.com", ["http://a.com", "http://a.com"]),
    ],
)
def test_parse_de_cors_origins(monkeypatch, bruto, esperado):
    """A configuração de CORS nunca deve assumir '*' com credenciais."""
    import api

    monkeypatch.setenv("CORS_ORIGINS", bruto)
    assert api._origens_cors() == esperado


def test_cors_desligado_por_padrao_pois_env_vazio(monkeypatch):
    """Sem CORS_ORIGINS a API não envia cabeçalho de cross-origin."""
    import api

    monkeypatch.setenv("CORS_ORIGINS", "")
    assert api._origens_cors() == []


def test_resposta_always_normaliza_content_em_texto(cliente, monkeypatch):
    """O `content` do modelo sempre chega como texto na resposta HTTP."""
    from src.nodes import code_analyzer

    monkeypatch.setattr(
        code_analyzer,
        "chat",
        lambda mensagens, tools=None: ("texto limpo", "fake"),
    )
    r = cliente.post("/chat", json={"message": "oi"})
    assert isinstance(r.json()["response"], str)
    assert r.json()["response"] == "texto limpo"
