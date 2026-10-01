"""Testes de `src/tools/observability.py`: isolamento por execução, latência e thread-safety."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.tools.observability import RunObserver, get_observer, run_scope


def _ler_jsonl(caminho):
    return [json.loads(linha) for linha in Path(caminho).read_text(encoding="utf-8").splitlines()]


class TestCicloDeVida:
    def test_start_run_gera_run_id_e_caminhos(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        run_id = observador.start_run(session_id="abc")

        assert run_id
        assert observador._jsonl_path.endswith(f"run_{run_id}.jsonl")
        assert observador._audit_path.endswith(f"audit_{run_id}.json")

    def test_run_start_e_gravado_no_jsonl(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        run_id = observador.start_run(session_id="abc")
        caminhos = observador.finish_run()

        eventos = _ler_jsonl(caminhos["structured_log"])
        inicio = next(e for e in eventos if e["event"] == "run_start")
        assert inicio["run_id"] == run_id
        assert inicio["session_id"] == "abc"

    def test_todos_os_eventos_carregam_o_mesmo_run_id(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        run_id = observador.start_run(session_id="abc")
        observador.node_started("responder_chat")
        observador.node_finished("responder_chat", 12.345, "ok")
        observador.llm_attempt("gemini", ok=True, duration_ms=5.0)
        caminhos = observador.finish_run()

        eventos = _ler_jsonl(caminhos["structured_log"])
        assert {e["run_id"] for e in eventos} == {run_id}

    def test_eventos_tem_timestamp_iso_utc(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run(session_id="abc")
        caminhos = observador.finish_run()

        for evento in _ler_jsonl(caminhos["structured_log"]):
            assert evento["ts"].endswith("+00:00")


class TestCorrelacaoDosDoisSinais:
    def test_auditoria_aponta_para_o_jsonl_da_mesma_execucao(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        run_id = observador.start_run(session_id="abc")
        observador.node_finished("responder_chat", 10.0, "ok")
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["run_id"] == run_id
        assert auditoria["artifacts"]["structured_log"].endswith(f"run_{run_id}.jsonl")
        # O caminho referenciado precisa existir de fato.
        assert Path(auditoria["artifacts"]["structured_log"]).exists()

    def test_auditoria_consolida_latencias(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        for latencia in (10.0, 20.0, 30.0):
            observador.node_finished("responder_chat", latencia, "ok")
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        no = auditoria["nodes_latency"]["responder_chat"]
        assert no["calls"] == 3
        assert no["min_ms"] == 10.0
        assert no["max_ms"] == 30.0
        assert no["avg_ms"] == 20.0

    def test_falha_de_no_marca_outcome_failed(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        observador.node_finished("responder_chat", 1.0, "error", error="falhou")
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["outcome"] == "failed"
        assert auditoria["nodes_with_errors"] == {"responder_chat": 1}

    def test_execucao_sem_erro_marca_outcome_succeeded(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        observador.node_finished("responder_chat", 1.0, "ok")
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["outcome"] == "succeeded"
        assert auditoria["status"] == "completed"

    def test_status_crashed_e_refletido(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        caminhos = observador.finish_run(status="crashed")

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["status"] == "crashed"


class TestFallbackDeLLM:
    def test_provedor_bem_sucedido_e_registrado(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        observador.llm_attempt("gemini", ok=True, duration_ms=8.0)
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["llm"]["providers_succeeded"] == ["gemini"]
        assert auditoria["llm"]["fallback_count"] == 0

    def test_fallback_conta_as_tentativas_falhas(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        observador.llm_attempt("gemini", ok=False, duration_ms=3.0, error="quota")
        observador.llm_attempt("groq", ok=True, duration_ms=5.0)
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["llm"]["fallback_count"] == 1
        assert auditoria["llm"]["providers_succeeded"] == ["groq"]
        assert auditoria["llm"]["failed_attempts"][0]["provider"] == "gemini"

    def test_erro_do_provider_e_truncado(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        observador.llm_attempt("gemini", ok=False, duration_ms=1.0, error="e" * 1000)
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert len(auditoria["llm"]["failed_attempts"][0]["error"]) == 200

    def test_sucesso_repetido_nao_duplica_no_lista(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        observador.llm_attempt("gemini", ok=True, duration_ms=1.0)
        observador.llm_attempt("gemini", ok=True, duration_ms=1.0)
        caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["llm"]["providers_succeeded"] == ["gemini"]


class TestRobustez:
    def test_finish_run_e_idempotente(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        primeiro = observador.finish_run()
        segundo = observador.finish_run()

        assert primeiro == segundo
        auditoria = json.loads(Path(primeiro["audit"]).read_text(encoding="utf-8"))
        assert auditoria["run_id"] == observador.run_id

    def test_eventos_pos_encerramento_sao_descartados(self, logs_dir):
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run()
        caminhos = observador.finish_run()
        observador.node_started("tardio")

        eventos = _ler_jsonl(caminhos["structured_log"])
        assert not [e for e in eventos if e["event"] == "node_start"]

    def test_escrita_em_diretorio_invalido_nao_propaga(self, tmp_path):
        """Best-effort: observabilidade jamais derruba a execução observada."""
        observador = RunObserver(logs_dir=str(tmp_path / "arquivo" / "sub"))
        (tmp_path / "arquivo").write_text("sou um arquivo, nao um diretorio")

        observador.start_run()  # não deve explodir
        observador.node_finished("n", 1.0, "ok")
        observador.finish_run()  # não deve explodir


class TestEscopoPorExecucao:
    def test_run_scope_registra_e_encerra(self, logs_dir):
        with run_scope(session_id="abc", logs_dir=logs_dir) as observador:
            run_id = observador.run_id
            observador.node_finished("responder_chat", 5.0, "ok")
            caminhos = observador.finish_run()

        auditoria = json.loads(Path(caminhos["audit"]).read_text(encoding="utf-8"))
        assert auditoria["run_id"] == run_id

    def test_run_scope_gera_auditoria_mesmo_sem_finish_explicito(self, logs_dir):
        with run_scope(session_id="abc", logs_dir=logs_dir) as observador:
            caminho_auditoria = observador._audit_path

        assert Path(caminho_auditoria).exists()

    def test_finish_explicito_nao_e_duplicado_pelo_scope(self, logs_dir):
        with run_scope(session_id="abc", logs_dir=logs_dir) as observador:
            observador.finish_run(status="crashed")

        auditoria = json.loads(Path(observador._audit_path).read_text(encoding="utf-8"))
        assert auditoria["status"] == "crashed"

    def test_execucoes_concorrentes_nao_compartilham_run_id(self, logs_dir):
        """Regressão: com singleton, requisições simultâneas sobrescreviam o run_id."""
        vistos = []
        trava = threading.Lock()

        def _worker(i):
            with run_scope(session_id=f"s{i}", logs_dir=logs_dir) as observador:
                observador.node_finished("responder_chat", float(i), "ok")
                with trava:
                    vistos.append(observador.run_id)
                observador.finish_run()

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_worker, range(8)))

        assert len(set(vistos)) == 8

    def test_requisicoes_paralelas_escrevem_em_arquivos_distintos(self, logs_dir):
        auditorias_por_thread = {}
        trava = threading.Lock()

        def _worker(i):
            with run_scope(session_id=f"s{i}", logs_dir=logs_dir) as observador:
                observador.node_finished("responder_chat", float(i), "ok")
                caminhos = observador.finish_run()
                with trava:
                    auditorias_por_thread[i] = caminhos["audit"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(_worker, range(6)))

        assert len(set(auditorias_por_thread.values())) == 6
        # Cada arquivo só pode conter o run da sua própria thread.
        for indice, caminho in auditorias_por_thread.items():
            auditoria = json.loads(Path(caminho).read_text(encoding="utf-8"))
            assert auditoria["session_id"] == f"s{indice}"


class TestGetObserver:
    def test_fora_de_run_scope_devolve_instancia_inerte(self):
        observador = get_observer()
        assert observador is get_observer()
        # Não lança nem escreve em disco.
        observador.node_started("x")
        observador.log_error("x", "y")

    def test_dentro_do_scope_devolve_a_execucao_corrente(self, logs_dir):
        with run_scope(session_id="abc", logs_dir=logs_dir) as escopo:
            assert get_observer() is escopo

    def test_apos_o_scope_volta_para_a_inerte(self, logs_dir):
        inerte = get_observer()
        with run_scope(session_id="abc", logs_dir=logs_dir):
            pass
        assert get_observer() is inerte


class TestThreadSafety:
    def test_escrita_concorrente_no_mesmo_jsonl(self, logs_dir):
        """Várias threads no MESMO observador não corrompem nem perdem o JSONL."""
        observador = RunObserver(logs_dir=logs_dir)
        observador.start_run(session_id="conc")

        def _worker(_):
            for _ in range(20):
                observador.log_event("node_start", node="responder_chat")

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(_worker, range(8)))

        caminhos = observador.finish_run()

        with open(caminhos["structured_log"], encoding="utf-8") as arquivo:
            registros = [json.loads(linha) for linha in arquivo if linha.strip()]

        # 1 run_start + 160 eventos concorrentes + 1 run_end, sem perda por corrida.
        assert all(r["run_id"] == observador.run_id for r in registros)
        assert len(registros) == 162
