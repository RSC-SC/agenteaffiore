"""Dublês de teste compartilhados: respostas e clientes de LLM controlados."""


class FakeResponse:
    """Resposta de LLM controlado por teste.

    Reproduz o formato que o LangChain devolve, incluindo `content` como lista
    de blocos — caso que o Gemini usa e que a normalização precisa cobrir.
    """

    def __init__(self, content="resposta", tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class FakeLLM:
    """ChatModel falso: devolve uma sequência de respostas ou exceções.

    `bind_tools` devolve `self`, de modo que as asserções sobre `calls` seguem
    válidas depois do vínculo de tools.
    """

    def __init__(self, respostas=None, nome="fake"):
        self.respostas = list(respostas or [FakeResponse("ok")])
        self.calls = []
        self.nome = nome
        self.tools_vinculadas = None

    def bind_tools(self, tools):
        self.tools_vinculadas = list(tools)
        return self

    def invoke(self, mensagens):
        self.calls.append(mensagens)
        if not self.respostas:
            return FakeResponse("resposta padrão")
        proximo = self.respostas.pop(0)
        if isinstance(proximo, Exception):
            raise proximo
        return proximo


class FakeTool:
    """Tool mínima: só precisa de `.name` e `.invoke`, como a BaseTool."""

    def __init__(self, nome, fn, descricao="tool de teste"):
        self.name = nome
        self.description = descricao
        self._fn = fn

    def invoke(self, args):
        return self._fn(args)


class ObjetoComStr:
    """Objeto sem `.content`, para testar a rede de segurança de `_texto`."""

    def __str__(self):
        return "bruto"


class ErroHttp(Exception):
    """Erro de API com status HTTP, como os SDKs da OpenAI e do Gemini levantam.

    Reproduz o caso real visto com a chave do OpenRouter: 403 por limite mensal
    excedido, que é o que o agente deve classificar como falha permanente de
    credencial (e não como algo que se resolve tentando de novo).
    """

    def __init__(self, status_code, mensagem="falha simulada"):
        super().__init__(f"Error code: {status_code} - {mensagem}")
        self.status_code = status_code
