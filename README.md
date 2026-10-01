# Agente de Chat Affiore

Assistente virtual da **Affiore (Arte em Presentear)**: atende clientes no
WhatsApp, tira dúvidas sobre os produtos, informa preços e envia os links dos
catálogos oficiais. Grafo **LangGraph** servido por **FastAPI**, com fallback
entre provedores de LLM.

```
Cliente (chat.html / n8n / WhatsApp)
        │  POST /chat
        ▼
     api.py ── valida, guarda memória da sessão, invoca o grafo
        │
        ▼
  src/graph.py ── grafo linear de 1 nó, instrumentado
        │
        ▼
 src/nodes/code_analyzer.py ── prompt da Affiore + formatação do estado
        │
        ├── src/tools/llm_tool.py ── Gemini → Groq → OpenRouter, com tool-calling
        │       └── src/tools/affiore_tool.py ── links e preços do catálogo
        └── src/tools/observability.py ── JSONL + auditoria por execução
```

---

## Início rápido

```bash
git clone https://github.com/RSC-SC/agenteaffiore
cd agenteaffiore
pip install -r requirements-dev.txt      # inclui testes e lint

cp .env.example .env                     # preencha ao menos uma chave de LLM
uvicorn api:app --reload --port 8000
```

- API: <http://localhost:8000> (Swagger em `/docs`)
- Health check: <http://localhost:8000/health>
- Interface web: abra `chat.html` no navegador e aponte a URL para o `/chat`

Sem `.env`, a API sobe normalmente — apenas o `/chat` responde
"Atendimento temporariamente indisponível", porque nenhum provedor está
configurado. Isso é proposital: o boot do serviço não depende de credencial.

### Pela CLI

```bash
python main.py
```

Conversa no terminal, sem subir a API. Útil para depurar o grafo e ajustar o
prompt.

### Com Docker

```bash
docker build -t affiore-chat .
docker run --rm -p 8000:8000 --env-file .env affiore-chat
```

A imagem roda como usuário não-root e traz `HEALTHCHECK` próprio.

---

## Configuração

Todas as variáveis estão documentadas e comentadas em [`.env.example`](.env.example).
As que mais importam:

| Variável | Padrão | Para quê |
|---|---|---|
| `GOOGLE_API_KEY` / `GROQ_API_KEY` / `OPENROUTER_API_KEY` | — | Chaves dos provedores. **Ao menos uma é obrigatória.** |
| `LLM_PRIMARY_PROVIDER` | `gemini` | Qual provedor é tentado primeiro (`gemini`, `groq`, `openrouter`) |
| `API_AUTH_TOKEN` | vazio | Se definido, `/chat` e `DELETE` exigem `Authorization: Bearer <token>` |
| `CORS_ORIGINS` | vazio | Origens liberadas. **Vazio = sem CORS** (só a própria origem) |
| `SESSION_TTL_MINUTES` | `60` | Aging da conversa |
| `SESSION_MAX` | `1000` | Teto de sessões simultâneas |
| `RATE_LIMIT_REQUESTS` | `30` | Requisições por janela de 60s, por cliente |
| `DISABLE_LLM` | `false` | Modo de contingência: uma mensagem padrão por sessão |

> **Antes de expor o serviço:** defina `API_AUTH_TOKEN`. Sem ele a API é aberta e
> cada chamada gasta tokens de LLM. O endpoint avisa isso no log no boot.

---

## Endpoints

### `POST /chat`

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_AUTH_TOKEN" \
  -d '{"message": "Quero um box de café para duas pessoas", "session_id": "cli-42"}'
```

```json
{
  "session_id": "cli-42",
  "response": "Que lovely! Para duas pessoas, o Box Café Seleto (R$ 89,00)...",
  "provider": null,
  "is_first_message": true,
  "error": null
}
```

| Campo | Descrição |
|---|---|
| `message` | Texto do cliente. 1 a 4000 caracteres. |
| `session_id` | Identificador da conversa (até 64 chars). Mantém a memória. |
| `metadata` | Objeto livre. Registrado no log, **não** vai para o LLM. |

Semântica de resposta:

- **200** com `error: null` — sucesso.
- **200** com `error` preenchida — falha de negócio (provedor indisponível, quota).
  É o que o n8n espera para poder encadear um fallback.
- **400** mensagem vazia · **401** credencial ausente ou inválida ·
  **422** payload fora do schema · **429** rate limit · **500** falha inesperada.

O corpo de um 500 **nunca** contém detalhe técnico do provedor.

### `DELETE /chat/{session_id}`

Limpa a memória de uma sessão. Devolve `{"session_id": ..., "removed": true|false}`.

### `GET /health`

```json
{"status": "ok", "service": "agente-chat-affiore", "sessoes": {"sessoes_ativas": 1, ...}}
```

---

## Fallback de provedores

O agente tenta **Gemini → Groq → OpenRouter** e avança quando o anterior falha
por quota, 429 ou timeout. O fallback acontece **na invocação**, não apenas na
construção do cliente: um provedor que responde com erro libera o próximo com a
mesma conversa.

`LLM_PRIMARY_PROVIDER` inverte a ordem sem desligar o fallback. Cada provedor
declara se suporta tool-calling — os modelos `:free` do OpenRouter não a suportam
de forma confiável, então são chamados **sem** tools e recebem os links e preços
do catálogo pelo prompt.

Os provedores usados, as tentativas que falharam e a contagem de fallbacks ficam
na auditoria de cada execução.

---

## Tool de catálogo

`obter_links_catalogo` devolve os links oficiais do Google Drive. Links e preços
vivem **apenas** em [`src/tools/affiore_tool.py`](src/tools/affiore_tool.py) —
é a fonte única, e tanto o prompt quanto a tool leem de lá. Para mudar um preço
ou um link, edite esse arquivo e nada mais.

O tool-calling é delimitado por `MAX_TOOL_ROUNDS`: um modelo que re-solicita a
tool indefinidamente não entra em laço.

---

## Observabilidade

Cada requisição produz dois artefatos correlacionados por um `run_id` único:

| Sinal | Arquivo | Conteúdo |
|---|---|---|
| Log estruturado (JSONL) | `logs/run_<run_id>.jsonl` | Um evento JSON por linha: `run_start`, `node_start`, `node_end` (+`duration_ms`), `error`, `llm_provider_success` / `llm_provider_result`, `run_end` |
| Auditoria (JSON) | `logs/audit_<run_id>.json` | Latência total e por nó (mín/média/máx), provedores usados, tentativas falhas, contagem de fallbacks, nós com erro, desfecho e o caminho do JSONL da mesma execução |

Cada requisição HTTP recebe o **seu próprio** `RunObserver`, guardado num
`ContextVar` por `run_scope()`. Isso é obrigatório: com um singleton de processo,
requisições simultâneas sobrescreveriam o `run_id` umas das outras e misturariam
os eventos de clientes diferentes no mesmo arquivo. Há teste de concorrência
cobrindo isso.

Garantias: escritas thread-safe, **best-effort** (falha de log jamais derruba a
atendimento) e sem segredos nos artefatos — o erro do provedor é truncado no
registro e nunca chega ao cliente.

---

## Segurança

- **Autenticação** por `API_AUTH_TOKEN`, em comparação de tempo constante.
- **CORS desligado por padrão.** Componha `CORS_ORIGINS` explicitamente; nunca
  use `*` com credenciais.
- **Rate limit** por cliente (janela deslizante) — é a proteção de custo, já que
  todo `/chat` consome tokens.
- **Nenhum detalhe interno na resposta.** Exceções do provedor viram log com
  traceback; o cliente recebe uma frase amigável.
- **Rate limit de tool-calling** (`MAX_TOOL_ROUNDS`) evita laço infinito com custo.
- **`.env` nunca é versionado**; só `.env.example` entra no repositório.
- **Container sem privilégios** (`USER affiore`).

A memória de sessão é **em processo**. Para múltiplas réplicas, substitua
`SessionStore` por um backend compartilhado (Redis) — a interface já está
delimitada no método `travar()`.

---

## Testes

**173 testes, 100% offline** — nenhum toca rede e nenhum exige credencial. É o
que permite rodar o CI sem expor segredo nenhum.

```bash
python -m pytest tests -q                                    # rápido
python -m pytest tests --cov=src --cov=api --cov-report=term-missing
python -m ruff check .
```

| Módulo | Foco |
|---|---|
| `test_api.py` | Endpoints, validação, sessão, rate limit, auth, concorrência, não-vazamento de erro |
| `test_session_store.py` | `SessionStore` (TTL, tetos, travas) e `RateLimiter` |
| `test_llm_tool.py` | Ordem de fallback, tool-calling, normalização de `content` |
| `test_code_analyzer.py` | Formatação de estado, prompt, modo de contingência |
| `test_graph.py` | Montagem do grafo, wrapper de instrumentação, E2E |
| `test_observability.py` | Correlação dos dois sinais, isolamento e thread-safety |
| `test_affiore_tool.py` | Links, preços e tools do catálogo |

Vários testes documentam **regressões concretas** que existiram no código — o
comentário do teste explica o sintoma que o observava. Os mais relevantes:

- o `content` do Gemini chegando como lista de blocos e virando lixo na resposta;
- `fn(state) or {}` no wrapper mascarando um nó que devolvia `None`;
- o `run_id` compartilhado entre requisições concorrentes;
- o TTL de sessão que nunca removia entrada alguma;
- a corrida de read-modify-write do histórico na mesma conversa.

---

## CI

Três jobs independentes em `.github/workflows/ci.yml`, todos sem credencial:

| Job | O que faz |
|---|---|
| `lint` | `ruff check .` |
| `test` | suíte offline com cobertura de `src` e `api.py` |
| `build` | `compileall`, smoke de importação do grafo e da API, e `docker build` — valida o `Dockerfile` |

Dispara em `push` para `main`/`develop` e em PR para ambos.

---

## Estrutura

```
agenteaffiore/
├── .env.example              # configuração documentada
├── .github/workflows/ci.yml   # lint + testes + build + docker build
├── AGENTS.md                 # instruções para agentes de código
├── Dockerfile
├── api.py                    # FastAPI: valida, sessão, invoca o grafo
├── chat.html                 # interface web de demonstração
├── main.py                   # CLI para depurar sem subir a API
├── pyproject.toml            # config do ruff
├── pytest.ini
├── requirements.txt          # produção
├── requirements-dev.txt      # + pytest, ruff, httpx
├── src/
│   ├── graph.py              # grafo LangGraph (1 nó) + instrumentação
│   ├── state.py              # ChatState (TypedDict, total=False)
│   ├── nodes/
│   │   └── code_analyzer.py  # prompt da Affiore + formatação de estado
│   └── tools/
│       ├── affiore_tool.py   # links, preços e tools do catálogo
│       ├── llm_tool.py       # provedores, fallback, tool-calling
│       └── observability.py  # JSONL + auditoria por execução
└── tests/                    # 173 testes offline
```

---

## Decisões técnicas

| Decisão | Por quê |
|---|---|
| LangGraph com **1 nó** | O grafo precisa orquestrar estado e instrumentação, não ramificar. Um `StateGraph` linear dá observabilidade e testabilidade sem arestas condicionais. |
| Fallback na **invocação** | Só na construção, um provedor que responde com quota derruba a conversa. O fallback real exige tentar a chamada. |
| `llm_tool` / `affiore_tool` como fonte única | Havia três fábricas de LLM com modelos default divergentes, o que tornava o comportamento dependente de qual módulo "vencesse". Uma fonte elimina a classe inteira do bug. |
| `total=False` sem reducer | O grafo é linear e não há escrita concorrente de chaves. Cada nó devolve `{**state, ...}` — devolver só a chave nova apagaria o resto. |
| `ContextVar` para observabilidade | Singleton de processo sobrescreve o `run_id` entre requisições simultâneas. |
| Trava por sessão | Atender uma mensagem é read-modify-write do histórico; sem serializar, duas requisições na mesma conversa se perdem. Sessões diferentes seguem em paralelo. |
| Erro de provedor vira log, não resposta | O técnico precisa do traceback; o cliente precisa de uma frase. Misturar os dois vaza endpoint e chave. |
| CORS desligado por padrão | O padrão navegador é a própria origem. Habilitar cross-origin é uma decisão deliberada, por variável de ambiente. |

---

## Limitações conhecidas

- **Memória de sessão em processo.** Reiniciar a API apaga as conversas; múltiplas
  réplicas não compartilham estado.
- **Rate limit por processo.** Com várias réplicas, o teto efetivo é o somatório delas.
- **Análise por prompt, sem recuperação.** As respostas vêm do prompt, não de uma
  base de conhecimento — o modelo é instruído a não inventar produto nem preço,
  mas não há verificação factual.
- **Sem streaming.** A resposta chega inteira; em integração real de WhatsApp
  vale avaliar streaming ou envio por webhook do n8n.
- **Preços no código.** Crítico para consistência, mas exige deploy para mudar
  qualquer valor. Catálogo próprio da loja exigiria uma fonte externa.

---

## Histórico

O repositório nasceu como *Agente Revisor de PRs* e foi convertido para o *Agente
de Chat Affiore* na [Issue #1](https://github.com/RSC-SC/agenteaffiore/issues/1).
O código do revisor, seus testes, o workflow n8n e a documentação correspondente
foram removidos — o escopo atual é exclusivamente o chat da Affiore.
