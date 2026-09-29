"""Nó responsável pelo atendimento e processamento de mensagens com LLM e Tools."""
import logging
from typing import Dict, Any, List
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

from src.tools.affiore_tool import AFFIORE_TOOLS
from src.tools.llm_tool import get_llm

logger = logging.getLogger(__name__)

# Mapeamento dinâmico para execução das tools
TOOL_MAP = {t.name: t for t in AFFIORE_TOOLS}

PROMPT_SISTEMA_AFFIORE = """Você é a atendente virtual da loja Affiore (Arte em Presentear).
Seu objetivo é atender clientes no WhatsApp de forma calorosa, acolhedora, educada e elegante.

Sobre a Affiore:
- Especializada em cestas de café da manhã, boxes de frios, vinhos, kits de spa e presentes corporativos.
- Criamos momentos em memórias afetivas.
- Contamos com curadoria de nutricionista (opções sem glúten e sem lactose disponíveis sob consulta prévia).
- Entregas: A taxa de entrega é calculada à parte de acordo com o endereço.

Produtos e Valores Oficiais:
- Box Café Seleto: R$ 89,00 + taxa (Drip coffee, suco integral, Slim Toast, manteiga, geleia 100% fruta, biscoito caseiro e mini bolo)
- Mini Box Frios: R$ 89,90 + taxa (Caixa acrílico, torradas, geleia, salame italiano, brie, gouda, damasco, azeitona, biscoito salgado, uvas)
- Box Afeto e Flores: R$ 209,00 + taxa (Buquê de flores, caneca personalizada com inicial, biscoitos artesanais em pote de vidro, chocolates Ferrero Rocher em caixa MDF)
- Kit Spa: R$ 259,00 + taxa (Geleia de banho, sais de banho, sabonete massageador, vela aromática hidratante, toalha de lavabo bordada/pintada à mão)
- Box Celebrar: R$ 359,00 + taxa (Cesta Pinus com alça de couro, 2 cervejas artesanais IPA Lagunitas, snacks, queijos, charcutaria e pães)
- Box Vinho Affiore: R$ 389,00 + taxa (Vinho tinto chileno Casillero del Diablo Reserva 750ml, queijos nobres, charcutaria, pães, nuts, geleia e Ferrero Rocher)
- Tábuas de Frios:
  * PP (20 cm) - R$ 159,00 (Serve 1 refeição ou 2 aperitivo)
  * P (25 cm) - R$ 199,00 (Serve 2 refeição ou 3 aperitivo)
  * M (30 cm) - R$ 279,00 (Serve 3 refeição ou 4 aperitivo)
  * G (35 cm) - R$ 359,00 (Serve 4 refeição ou 4-5 aperitivo)

Personalizações e Adicionais:
- Cartão com mensagem dedicada: CORTESIA
- Mini Buquê Astromélia: R$ 30,00 | Flores Secas: R$ 40,00 | Orquídea: R$ 99,00
- Caneca personalizada: R$ 40,00 | Foto Polaroid: R$ 14,00
- Balão Bubble: R$ 45,00 | Balão simples (5 un): R$ 20,00 | Bento Cake: R$ 85,00

Instruções Operacionais:
1. Se o cliente pedir fotos, catálogos, cardápio ou quiser ver todas as opções disponíveis, ACIONE IMEDIATAMENTE a ferramenta `obter_links_catalogo` para fornecer os links oficiais do Google Drive.
2. Pergunte sempre para quando é a ocasião e ajude a indicar o presente mais adequado.
3. Mantenha respostas claras, pontuais e bem formatadas para leitura no WhatsApp.
"""


def _converter_mensagens(historico: List[Any]) -> List[BaseMessage]:
    """Converte mensagens do formato dicionário ou objetos LangChain para BaseMessage."""
    convertidas = []
    for msg in historico:
        if isinstance(msg, BaseMessage):
            convertidas.append(msg)
        elif isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ["user", "human"]:
                convertidas.append(HumanMessage(content=content))
            elif role in ["assistant", "ai"]:
                convertidas.append(AIMessage(content=content))
            elif role == "system":
                convertidas.append(SystemMessage(content=content))
    return convertidas


def consultar_llm(state: Dict[str, Any]) -> Dict[str, Any]:
    """Executa o ciclo de consulta à LLM e resolve eventuais chamadas de ferramentas."""
    raw_messages = state.get("messages", [])
    mensagens_langchain = _converter_mensagens(raw_messages)

    # Injeta a instrução de sistema no início
    chat_input = [SystemMessage(content=PROMPT_SISTEMA_AFFIORE)] + mensagens_langchain

    llm = get_llm()
    resposta = llm.invoke(chat_input)

    # Verifica se a LLM solicitou a execução de alguma ferramenta
    if hasattr(resposta, "tool_calls") and resposta.tool_calls:
        mensagens_com_tools = list(chat_input) + [resposta]

        for call in resposta.tool_calls:
            tool_name = call.get("name")
            tool_args = call.get("args", {})
            tool_id = call.get("id")

            tool_func = TOOL_MAP.get(tool_name)
            if tool_func:
                try:
                    resultado_tool = tool_func.invoke(tool_args)
                except Exception as e:
                    logger.error(f"Erro ao executar tool {tool_name}: {e}")
                    resultado_tool = f"Erro ao acessar informações do catálogo: {e}"

                mensagens_com_tools.append(
                    ToolMessage(
                        tool_call_id=tool_id,
                        content=str(resultado_tool),
                        name=tool_name
                    )
                )

        # Segunda chamada para gerar a resposta final com os dados da ferramenta
        resposta_final = llm.invoke(mensagens_com_tools)
        texto_resposta = resposta_final.content
    else:
        texto_resposta = resposta.content

    # Retorna o histórico atualizado e a resposta textual direta
    mensagens_atualizadas = list(raw_messages) + [
        {"role": "assistant", "content": texto_resposta}
    ]

    return {
        **state,
        "messages": mensagens_atualizadas,
        "response": texto_resposta
    }