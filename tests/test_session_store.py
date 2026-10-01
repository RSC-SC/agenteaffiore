"""Testes de `api.py` — SessionStore e RateLimiter (unidades, sem HTTP)."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import pytest


@pytest.fixture
def store():
    import api

    return api.SessionStore(ttl_minutes=60, max_sessions=100, max_messages=10)


class TestSessionStore:
    def test_primeiro_toque_marca_sessao_nova(self, store):
        assert store.touch("a") is True

    def test_segundo_toque_nao_e_sessao_nova(self, store):
        store.touch("a")
        assert store.touch("a") is False

    def test_sessoes_distintas_sao_independentes(self, store):
        store.touch("a")
        assert store.touch("b") is True

    def test_historico_comeca_vazio(self, store):
        store.touch("a")
        assert store.history("a") == []

    def test_historico_de_sessao_inexistente_e_vazio(self, store):
        assert store.history("nao-existe") == []

    def test_historico_persiste(self, store):
        store.touch("a")
        store.set_history("a", [{"role": "user", "content": "oi"}])
        assert store.history("a") == [{"role": "user", "content": "oi"}]

    def test_historico_e_copia_defensiva(self, store):
        """O chamador não pode mutar o estado interno sem passar por set_history."""
        store.touch("a")
        store.set_history("a", [{"role": "user", "content": "oi"}])
        devolvido = store.history("a")
        devolvido.append({"role": "user", "content": "injetado"})
        assert len(store.history("a")) == 1

    def test_historico_e_limitado(self, store):
        store.touch("a")
        for i in range(50):
            store.set_history("a", [{"role": "user", "content": str(i)}])
        assert len(store.history("a")) <= 10

    def test_limite_mantem_as_mensagens_mais_recentes(self, store):
        store.touch("a")
        for i in range(15):
            store.set_history("a", [{"role": "user", "content": str(i)}])
        assert store.history("a")[-1]["content"] == "14"

    def test_expira_sessao_antiga(self, store):
        """Regressão: antes o TTL só era consultado na leitura do mesmo id,
        então entradas nunca eram removidas."""
        store.touch("velha")
        store.set_history("velha", [{"role": "user", "content": "oi"}])

        # Finge que a sessão foi vista há 2 horas.
        store._sessoes["velha"] = (datetime.now() - timedelta(hours=2), [])

        assert store.touch("nova") is True
        # 'velha' foi removida, só 'nova' permanece.
        assert store.stats()["sessoes_ativas"] == 1
        assert "velha" not in store._sessoes
        assert "nova" in store._sessoes

    def test_reset_remove_a_sessao(self, store):
        store.touch("a")
        assert store.reset("a") is True
        assert store.history("a") == []

    def test_reset_de_sessao_inexistente(self, store):
        assert store.reset("nao-existe") is False

    def test_teto_de_capacidade_e_respeitado(self, store):
        for i in range(150):
            store.touch(f"s{i}")
        assert store.stats()["sessoes_ativas"] <= 100

    def test_eviccao_descarta_as_mais_antigas(self, store):
        for i in range(150):
            store.touch(f"s{i}")
        assert "s0" not in store._sessoes
        assert "s149" in store._sessoes

    def test_stats_reporta_a_politica(self, store):
        stats = store.stats()
        assert stats == {"sessoes_ativas": 0, "max_sessoes": 100, "ttl_minutes": 60}

    def test_acesso_concorrente_nao_corrompe(self, store):
        """Regressão: o dict era mutado sem lock, perdendo mensagens."""
        erros = []

        def _worker(i):
            try:
                for _ in range(20):
                    store.touch("comum")
                    atual = store.history("comum")
                    store.set_history("comum", [*atual, {"role": "user", "content": f"{i}"}])
            except Exception as exc:  # pragma: no cover
                erros.append(exc)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_worker, range(8)))

        assert not erros
        assert len(store.history("comum")) <= 10  # teto respeitado, sem estouro


class TestRateLimiter:
    def test_permite_ate_o_teto(self):
        import api

        limitador = api.RateLimiter(limite=3, janela_seg=60)
        assert [limitador.permitir("ip") for _ in range(3)] == [True, True, True]

    def test_bloqueia_apos_o_teto(self):
        import api

        limitador = api.RateLimiter(limite=2, janela_seg=60)
        limitador.permitir("ip")
        limitador.permitir("ip")
        assert limitador.permitir("ip") is False

    def test_clientes_distintos_tem_cotas_independentes(self):
        import api

        limitador = api.RateLimiter(limite=1, janela_seg=60)
        assert limitador.permitir("ip-a") is True
        assert limitador.permitir("ip-b") is True
        assert limitador.permitir("ip-a") is False

    def test_janela_desliza_e_libera_novamente(self):
        import api

        limitador = api.RateLimiter(limite=1, janela_seg=0.05)
        assert limitador.permitir("ip") is True
        assert limitador.permitir("ip") is False
        time.sleep(0.08)
        assert limitador.permitir("ip") is True

    def test_acesso_concorrente_nao_estoura_o_teto(self):
        """Regressão: a checagem e o registro precisam ocorrer sob o mesmo lock."""
        import api

        limitador = api.RateLimiter(limite=10, janela_seg=60)
        permitidos = []
        trava = threading.Lock()

        def _worker(_):
            ok = limitador.permitir("ip")
            with trava:
                permitidos.append(ok)

        with ThreadPoolExecutor(max_workers=20) as pool:
            list(pool.map(_worker, range(50)))

        assert sum(permitidos) == 10

    def test_chaves_distintas_nao_crescem_sem_limite(self):
        import api

        limitador = api.RateLimiter(limite=1000, janela_seg=60)
        for i in range(10_050):
            limitador.permitir(f"ip-{i}")
        assert len(limitador._eventos) <= 10_000


class TestTravaPorSessao:
    def test_serializa_a_mesma_sessao(self):
        """Regressão: sem a trava, duas requisições na mesma conversa fazem
        read-modify-write do histórico e a última gravação descarta a outra."""
        import api

        store = api.SessionStore(ttl_minutes=60, max_sessions=100, max_messages=100)
        ordem = []

        def _worker(i):
            with store.travar("mesma"):
                historico = store.history("mesma")
                store.set_history("mesma", [*historico, {"role": "user", "content": str(i)}])
                ordem.append(i)

        with ThreadPoolExecutor(max_workers=5) as pool:
            list(pool.map(_worker, range(20)))

        assert len(store.history("mesma")) == 20

    def test_nao_serializa_sessoes_diferentes(self):
        """Sessões distintas precisam continuar em paralelo."""
        import api

        store = api.SessionStore(ttl_minutes=60, max_sessions=100, max_messages=100)
        barreira = threading.Barrier(4, timeout=5)
        erros = []

        def _worker(i):
            try:
                with store.travar(f"s{i}"):
                    # Só passa se as quatro travas estiverem abertas ao mesmo tempo.
                    barreira.wait()
            except threading.BrokenBarrierError:
                erros.append(i)

        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(_worker, range(4)))

        assert not erros

    def test_trava_e_liberada_em_caso_de_excecao(self):
        import api

        store = api.SessionStore(ttl_minutes=60, max_sessions=100, max_messages=10)
        with pytest.raises(RuntimeError):
            with store.travar("s"):
                raise RuntimeError("falha no meio")

        # Se a trava tivesse vazado, esta aquisição travaria para sempre.
        with store.travar("s"):
            pass

    def test_travas_descartam_as_sessoes_mais_antigas(self):
        import api

        store = api.SessionStore(ttl_minutes=60, max_sessions=5, max_messages=10)
        for i in range(20):
            with store.travar(f"s{i}"):
                pass
        assert len(store._trancas) <= 5


class TestComparacaoDeSegredos:
    def test_token_correto_passa(self):
        import api

        assert api.secrets_compare("segredo", "segredo") is True

    def test_token_errado_falha(self):
        import api

        assert api.secrets_compare("errado", "segredo") is False

    def test_vazio_nunca_passa(self):
        """Token ausente não pode ser interpretado como credencial válida."""
        import api

        assert api.secrets_compare("", "segredo") is False
        assert api.secrets_compare("", "") is False
