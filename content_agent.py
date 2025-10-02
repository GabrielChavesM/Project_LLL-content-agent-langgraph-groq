import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# Importa as keys do dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# --- Configuração do LLM (MODELO MUDADO PARA GEMMA 2 9B - 15K TPM) ---
llm = ChatGroq(
    model_name="gemma2-9b-it",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

# ====================================================================
# I. DEFINIÇÃO DO ESTADO DO GRAFO (O "Contexto" Partilhado)
# ====================================================================

class ContentAgentState(TypedDict):
    """
    Representa o estado que é passado entre os nós do LangGraph.
    """
    topic: str
    outline: Annotated[str, operator.add] 
    content: Annotated[str, operator.add] 
    revision_needed: bool 
    review_feedback: str 

# ====================================================================
# II. DEFINIÇÃO DOS NÓS (As Funções de Ação)
# ====================================================================

# 1. NÓ: PLANEAMENTO (Gerar o Esboço)
def plan_content(state: ContentAgentState):
    """Gera o esboço (outline) do artigo com base no tópico fornecido."""
    print("--- 1. EXECUTANDO PLANEAMENTO: Criando Esboço ---")
    topic = state["topic"]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Você é um planeador de conteúdo de primeira linha. Sua tarefa é criar um esboço (outline) detalhado em markdown (usando ## para seções) para um artigo sobre o tópico fornecido. O esboço deve ser completo e lógico."),
        HumanMessage(content=f"Tópico para o artigo: {topic}")
    ])
    
    chain = prompt | llm
    outline_result = chain.invoke({"topic": topic}).content
    
    return {"outline": outline_result, "content": "", "review_feedback": ""} 

# 2. NÓ: REDAÇÃO (Gerar ou Revisar o Conteúdo - OTIMIZADO PARA MENOS TOKENS)
def draft_content(state: ContentAgentState):
    """Redige o conteúdo completo do artigo com base no esboço (outline), ou revisa-o se houver feedback."""
    topic = state["topic"]
    outline = state["outline"]
    feedback = state.get("review_feedback", "")
    
    is_redraft = bool(feedback) and feedback.strip().upper() not in ["OK", ""]
    
    if is_redraft:
        print("--- 2b. EXECUTANDO REDAÇÃO: Revisando o Rascunho com Feedback ---")
        
        # OTIMIZAÇÃO DE TOKENS: Não passamos o conteúdo anterior.
        # Pedimos ao LLM para reescrever o artigo completo apenas com base no esboço e no feedback.
        system_content = f"""
Você é um redator profissional altamente detalhista. Sua tarefa é REESCREVER O ARTIGO COMPLETO do zero, seguindo rigorosamente o esboço fornecido e resolvendo *todos* os problemas listados no feedback crítico.

Preste atenção especial a:
- Acentuação correta e gramática impecável em português.
- Completude do conteúdo, garantindo que todas as seções do esboço sejam desenvolvidas.
- Tom informativo, amigável e profissional.
"""
        user_content = f"""
TÓPICO: {topic}
ESBOÇO A SEGUIR:
{outline}

FEEDBACK DE REVISÃO (Obrigatório resolver estes pontos na nova redação):
{feedback}

REESCREVA O ARTIGO COMPLETO E FINALIZADO AQUI.
"""
    
    else:
        print("--- 2a. EXECUTANDO REDAÇÃO: Gerando Rascunho Inicial ---")
        system_content = "Você é um redator profissional. Sua tarefa é escrever o conteúdo COMPLETO do artigo, seguindo rigorosamente o esboço fornecido. Mantenha um tom informativo e amigável, mas profissional. NÃO use marcadores de posição ou frases introdutórias vazias."
        user_content = f"Tópico: {topic}\n\nEsboço para Redação:\n{outline}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ])
    
    chain = prompt | llm
    content_result = chain.invoke({}).content
    
    return {"content": content_result, "revision_needed": False, "review_feedback": ""} 

# 3. NÓ: REVISÃO (NÓ CONDICIONAL - O Coração do LangGraph)
def review_content(state: ContentAgentState):
    """
    Avalia o conteúdo gerado. Se a avaliação falhar (ex: tom incorreto, falta de detalhes),
    define 'revision_needed' como True e armazena o feedback detalhado.
    """
    print("--- 3. EXECUTANDO REVISÃO: Avaliando Qualidade e Tom ---")
    content = state["content"]
    
    # Prompt Modificado para forçar o LLM a dar feedback detalhado em caso de falha.
    review_prompt = f"""
    Avalie o seguinte rascunho do artigo. Você é um editor de conteúdo altamente rigoroso e metódico.
    
    Critérios de Reprovação (Falha se qualquer um for verdadeiro):
    1. Conteúdo Incompleto: O artigo termina abruptamente ou falta alguma seção prometida pelo esboço?
    2. Gramática e Ortografia: Existem erros gramaticais, ortográficos (incluindo acentuação errada, como 'Ä' em vez de 'Ã' ou 'Á') ou pontuação óbvios?
    3. Tom Inadequado: O tom está excessivamente formal/robótico, ou inadequado para um artigo técnico, mas amigável?
    
    INSTRUÇÃO DE SAÍDA:
    - Se o artigo for satisfatório e pronto para publicação, responda APENAS com a palavra "OK".
    - Se o artigo FALHAR em qualquer critério, responda EXCLUSIVAMENTE com o prefixo "REVISÃO:", seguido de dois pontos (:) e uma lista CLARA e numerada dos problemas encontrados e o que precisa ser corrigido. Exemplo: 'REVISÃO: 1. O artigo falha em ... 2. Corrija o erro 'Ä' para 'Ã'.'

    ARTIGO A AVALIAR (Primeiros 1000 caracteres):\n---\n{content[:1000]}...
    """
    
    # O LLM faz a avaliação
    review_chain = ChatPromptTemplate.from_template(review_prompt) | llm
    full_review_result = review_chain.invoke({}).content.strip()

    # Verifica se a resposta contém o prefixo de revisão
    result_keyword = full_review_result.split(':')[0].strip().upper()
    
    print(f"Resultado da Avaliação: {result_keyword}")
    
    if "REVISÃO" in result_keyword:
        print("!! REVISÃO NECESSÁRIA: Voltando ao nó de Redação com feedback.")
        # Retorna o feedback completo (ex: "REVISÃO: 1. Incompleto. 2. Erro de gramática.")
        return {"revision_needed": True, "review_feedback": full_review_result}
    else:
        print("✅ APROVADO: Conteúdo pronto para publicação.")
        return {"revision_needed": False, "review_feedback": "OK"} # Garante que o campo existe

# ====================================================================
# III. DEFINIÇÃO DA LÓGICA DO GRAFO
# ====================================================================

def should_continue(state: ContentAgentState):
    """
    Função de Roteamento Condicional.
    Define se o Agente deve continuar o ciclo (revisão falhou) ou terminar (revisão bem-sucedida).
    """
    if state["revision_needed"]:
        # Se precisar de revisão, volta para o nó de Redação
        return "re_draft"
    else:
        # Se estiver OK, termina a execução
        return "publish"

def create_agent_graph():
    """Constrói e compila o LangGraph."""
    
    # 1. Cria a instância do StateGraph
    workflow = StateGraph(ContentAgentState)

    # 2. Adiciona os Nós (as funções de ação)
    workflow.add_node("plan", plan_content)
    workflow.add_node("draft", draft_content)
    workflow.add_node("review", review_content)

    # 3. Define as Edges (transições)

    # 3a. Caminho inicial: Começa no Planeamento, vai para a Redação
    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "draft")

    # 3b. Transição de Redação para Revisão (sempre)
    workflow.add_edge("draft", "review")
    
    # 3c. Transição de Revisão (O Roteamento Condicional)
    workflow.add_conditional_edges(
        "review",        # Nó de origem
        should_continue, # Função de decisão (chama should_continue)
        {
            "re_draft": "draft", # Se a decisão for "re_draft", volta para o nó 'draft'
            "publish": END,      # Se a decisão for "publish", termina o LangGraph
        },
    )

    # 4. Compila o grafo para execução
    app = workflow.compile()
    return app

# ====================================================================
# IV. EXECUÇÃO DO AGENTE
# ====================================================================

if __name__ == "__main__":
    # CONFIGURAÇÃO DE AMBIENTE INTERNA (HARDCODING)
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "Content-Generator-Agent") 
    
    if not LANGCHAIN_API_KEY or not GROQ_API_KEY:
        print("ERRO: As chaves GROQ_API_KEY e LANGCHAIN_API_KEY devem ser definidas no código.")
    else:
        app = create_agent_graph()
        
        # Permite ao utilizador inserir o tópico na execução
        initial_topic = input("Insira o tópico para o artigo (ou deixe vazio para sair): ").strip()

        if not initial_topic:
            print("Nenhum tópico fornecido. A sair.")
            exit()

        print(f"\n>>>> INICIANDO AGENTE PARA O TÓPICO: {initial_topic} <<<<\n")

        # Executa o Agente (sincronamente neste exemplo)
        final_state = app.invoke({"topic": initial_topic})
        
        final_content = final_state["content"]

        print("\n\n#####################################################")
        print("########## AGENTE CONCLUÍDO. CONTEÚDO FINAL ###########")
        print("#####################################################")
        print(final_content)