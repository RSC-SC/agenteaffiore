# AGENTS.md — Instruções para o Agente IA

> Contexto do projeto, decisões e fluxo de trabalho para agentes de código.
> Lido automaticamente no início de cada sessão.

## O que é este projeto

**Agente de Chat Affiore** — assistente virtual da Affiore (Arte em Presentear),
que atende clientes e envia os links do catálogo. Grafo LangGraph + FastAPI.

Servido por `api.py` (HTTP) ou `main.py` (CLI). Interface de demonstração em
`chat.html`.

## Regras invioláveis

1. **Nunca versionar segredos.** `.env` está no `.gitignore`; só `.env.example`
   entra no repositório.
2. **A fonte única de cada dado é um só arquivo.** Preços e links do catálogo
   vivem só em `src/tools/affiore_tool.py`. Provedores e modelos de LLM vivem só
   em `src/tools/llm_tool.py`. Não redefina nem duplique.
3. **O que chega ao cliente nunca carrega detalhe interno.** Erros de provedor
   viram mensagem amigável; o técnico vai para o log. Não há `str(e)` na
   resposta HTTP.
4. **Tool-calling é delimitado.** `MAX_TOOL_ROUNDS` impede laço infinito.
5. **A suíte é 100% offline.** Nenhum teste toca rede ou exige credencial — é o
   que permite rodar o CI sem segredos.
6. **Nada de `except: pass` silencioso.** Falha de escrita de log é
   `logger.warning(..., exc_info=True)`, nunca descarte.

## Fluxo de desenvolvimento (obrigatório)

1. **Criar Issue no GitHub** — título, descrição clara, labels.
2. **Branch a partir de `develop`** — `feature/`, `fix/` ou `docs/`.
3. **Implementar** na branch criada.
4. **Commit** — mensagens semânticas (`feat:`, `fix:`, `refactor:`, `docs:`,
   `chore:`, `test:`).
5. **PR para `develop`** — referenciando a Issue (`Closes #N`) e descrevendo as
   mudanças.
6. **Atualizar a Issue** com o link do PR e o status.
7. **Integração** — `feature/*` → `develop` → `main`.

## Portões de qualidade (o CI falha se qualquer um falhar)

```bash
python -m pytest tests -q --cov=src --cov=api   # 173 testes, offline
python -m ruff check .                           # precisa zerar
python -m compileall -q src api.py main.py
docker build -t affiore-chat:ci .               # valida o Dockerfile
```

## Arquitetura

```
api.py                      FastAPI: valida, gerencia sessão, invoca o grafo
 └─ src/graph.py            Grafo LangGraph (1 nó) + wrapper de observabilidade
     └─ src/nodes/code_analyzer.py   Prompt da Affiore + formatação de estado
         ├─ src/tools/llm_tool.py     Provedores, fallback, tool-calling
         │   └─ src/tools/affiore_tool.py   Links, preços e tools do catálogo
         └─ src/tools/observability.py     JSONL + auditoria por execução
main.py                     CLI para depurar sem subir a API
chat.html                   Interface web de demonstração
tests/                      Suíte offline (unitário, integração e concorrência)
```

## Estado compartilhado (`src/state.py`)

`ChatState` é `TypedDict(total=False)` — todas as chaves opcionais porque o
LangGraph exige. O grafo é linear e **não usa reducers**, então cada nó devolve o
dicionário completo com `{**state, ...}`. Sem reducer, devolver só a chave nova
apagaria o resto do estado.

## Observabilidade

Cada execução recebe seu próprio `RunObserver`, guardado num `ContextVar` via
`run_scope()`. Isso é obrigatório: com um singleton de processo, requisições
HTTP simultâneas sobrescrevem o `run_id` uma da outra e misturam os eventos de
clientes diferentes no mesmo arquivo. Dois artefatos por execução, correlacionados
pelo `run_id`: `logs/run_<id>.jsonl` e `logs/audit_<id>.json`.

## Memória de sessão

`SessionStore` guarda as conversas em memória: TTL real com remoção na escrita,
teto de sessões e de mensagens. O ciclo de vida de uma conversa roda sob
`with sessoes.travar(session_id):` — atender uma mensagem é read-modify-write do
histórico e, sem serialização, duas requisições simultâneas na mesma conversa se
sobrescrevem. Sessões diferentes continuam em paralelo.

## Contexto do repositório

| Item | Valor |
|------|-------|
| Repositório | https://github.com/RSC-SC/agenteaffiore |
| Stack | LangGraph, FastAPI, LangChain, Pytest, Ruff, Docker |
| Python | 3.10+ (CI e imagem em 3.11) |
| Fluxo Git | `main` ← `develop` ← `feature/*` |

### Histórico relevante

Este repositório nasceu como *Agente Revisor de PRs* e foi convertido para o
*Agente de Chat Affiore* (Issue #1). Todo o código do revisor foi removido junto
com os testes, o workflow n8n e a documentação correspondentes. Não reintroduza
nós de revisão de PR — o escopo é exclusivamente o chat da Affiore.