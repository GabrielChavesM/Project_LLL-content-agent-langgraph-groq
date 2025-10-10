import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# --- CrewAI Imports ---
from crewai import Task  # Apenas Task, Agents serão simulados

# ====================================================================
# 0. CARREGAR VARIÁVEIS DE AMBIENTE
# ====================================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# ====================================================================
# I. CONFIGURAÇÃO DO LLM (LangChain)
# ====================================================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

# ====================================================================
# II. DEFINIÇÃO DO ESTADO DO GRAFO
# ====================================================================

class ContentAgentState(TypedDict):
    topic: str
    outline: Annotated[str, operator.add]
    content: Annotated[str, operator.add]
    revision_needed: bool
    review_feedback: str
    revision_count: int

# ====================================================================
# III. DEFINIÇÃO DOS NÓS DO LANGGRAPH
# ====================================================================

def plan_content(state: ContentAgentState):
    print("--- 1. EXECUTANDO PLANEAMENTO: Criando Esboço ---")
    topic = state["topic"]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Você é um planeador de conteúdo de primeira linha. Crie um esboço detalhado em markdown (usando ## para seções) para um artigo sobre o tópico fornecido."),
        HumanMessage(content=f"Tópico: {topic}")
    ])

    chain = prompt | llm
    outline_result = chain.invoke({"topic": topic}).content

    return {"outline": outline_result, "content": "", "review_feedback": "", "revision_count": 0}

def draft_content(state: ContentAgentState):
    topic = state["topic"]
    outline = state["outline"]
    feedback = state.get("review_feedback", "")

    revision_count = state.get("revision_count", 0) + 1
    is_redraft = bool(feedback) and feedback.strip().upper() not in ["OK", ""]

    if is_redraft:
        print(f"--- 2b. EXECUTANDO REDAÇÃO: Revisando o Rascunho (Revisão {revision_count}) ---")
        system_content = """
Você é um redator profissional. Reescreva o artigo do zero, seguindo o esboço e resolvendo todos os pontos do feedback.
"""
        user_content = f"Tópico: {topic}\n\nEsboço:\n{outline}\n\nFeedback:\n{feedback}\n\nReescreva o artigo completo."
    else:
        print("--- 2a. EXECUTANDO REDAÇÃO: Gerando Rascunho Inicial ---")
        system_content = "Você é um redator profissional. Redija o conteúdo completo do artigo com base no esboço."
        user_content = f"Tópico: {topic}\n\nEsboço:\n{outline}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ])

    chain = prompt | llm
    content_result = chain.invoke({}).content

    return {
        "content": content_result,
        "revision_needed": is_redraft,
        "review_feedback": feedback,
        "revision_count": revision_count
    }

# ====================================================================
# IV. INTEGRAÇÃO COM CREWAI USANDO GROQ (MULTI-AGENT, ITERATIVO, FEEDBACK LOOP)
# ====================================================================

class GroqAgentWrapper:
    """Wrapper para executar tarefas do CrewAI usando ChatGroq"""
    def __init__(self, llm):
        self.llm = llm

    def execute_task(self, task: Task):
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um redator e revisor profissional."),
            HumanMessage(content=task.description)
        ])
        chain = prompt | self.llm
        result = chain.invoke({})
        return getattr(result, "content", "")


def crew_review_content(state: ContentAgentState):
    print("--- 3. EXECUTANDO REVISÃO COM CREWAI (USANDO GROQ, MULTI-AGENT) ---")

    content = state["content"]
    groq_agent = GroqAgentWrapper(llm)

    for iteration in range(3):  # Fluxo iterativo até 3 revisões
        print(f"🔄 Iteração {iteration+1} da revisão")

        # Tarefas simuladas do CrewAI
        review_task = Task(
            description=f"Revise tecnicamente o artigo abaixo e liste melhorias detalhadas:\n\n{content}",
            expected_output="Lista de sugestões técnicas e de clareza.",
            agent=None
        )

        edit_task = Task(
            description="Reescreva o artigo usando o feedback do revisor, melhorando estilo, clareza e gramática.",
            expected_output="Nova versão do artigo revisada.",
            agent=None
        )

        # Executa tarefas sequencialmente
        review_result = groq_agent.execute_task(review_task)
        edit_task.description += f"\n\nFEEDBACK DO REVISOR:\n{review_result}"
        final_result = groq_agent.execute_task(edit_task)

        # Logging do processo
        print("🔍 Feedback do revisor:", review_result[:300], "...")
        print("✏️ Resultado da edição:", final_result[:300], "...")

        # Atualiza o conteúdo para próxima iteração
        content = final_result

        if "MELHORAR" not in final_result.upper() and "REVISÃO" not in final_result.upper():
            print("✅ Conteúdo aprovado sem necessidade de revisão adicional.")
            return {"revision_needed": False, "review_feedback": "OK", "content": final_result}

    print("⚠️ Revisão necessária após 3 iterações de CrewAI (Groq).")
    return {"revision_needed": True, "review_feedback": final_result, "content": final_result}

# ====================================================================
# V. LÓGICA DO GRAFO
# ====================================================================

def should_continue(state: ContentAgentState):
    if state["revision_needed"] and state.get("revision_count", 0) < 5:
        return "re_draft"
    else:
        return "publish"

def create_agent_graph():
    workflow = StateGraph(ContentAgentState)

    workflow.add_node("plan", plan_content)
    workflow.add_node("draft", draft_content)
    workflow.add_node("crew_review", crew_review_content)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "draft")
    workflow.add_edge("draft", "crew_review")
    workflow.add_conditional_edges(
        "crew_review",
        should_continue,
        {"re_draft": "draft", "publish": END}
    )

    return workflow.compile()

# ====================================================================
# VI. EXECUÇÃO FINAL
# ====================================================================

if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "LangChain-CrewAI-ArticleGen")

    if not LANGCHAIN_API_KEY or not GROQ_API_KEY:
        print("ERRO: Chaves LANGCHAIN_API_KEY e GROQ_API_KEY são necessárias.")
    else:
        app = create_agent_graph()
        initial_topic = input("Insira o tópico para o artigo (ex: Diferença entre LangChain e CrewAI): ").strip()
        if not initial_topic:
            print("Nenhum tópico fornecido. Encerrando.")
            exit()

        print(f"\n>>>> INICIANDO AGENTE PARA O TÓPICO: {initial_topic} <<<<\n")
        final_state = app.invoke({"topic": initial_topic})
        final_content = final_state.get("content", "")

        print("\n\n#####################################################")
        print("########## CONTEÚDO FINAL GERADO PELO AGENTE #########")
        print("#####################################################")
        print(final_content)
