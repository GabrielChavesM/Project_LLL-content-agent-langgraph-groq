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

# --- Configuração do LLM ---
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

# ====================================================================
# I. DEFINIÇÃO DO ESTADO DO GRAFO
# ====================================================================

class ContentAgentState(TypedDict):
    topic: str
    outline: Annotated[str, operator.add] 
    content: Annotated[str, operator.add] 
    revision_needed: bool 
    review_feedback: str 
    revision_count: int  # Contador de revisões

# ====================================================================
# II. DEFINIÇÃO DOS NÓS
# ====================================================================

def plan_content(state: ContentAgentState):
    print("--- 1. EXECUTANDO PLANEAMENTO: Criando Esboço ---")
    topic = state["topic"]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Você é um planeador de conteúdo de primeira linha. Sua tarefa é criar um esboço (outline) detalhado em markdown (usando ## para seções) para um artigo sobre o tópico fornecido. O esboço deve ser completo e lógico."),
        HumanMessage(content=f"Tópico para o artigo: {topic}")
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
        system_content = f"""
Você é um redator profissional altamente detalhista. Sua tarefa é REESCREVER O ARTIGO COMPLETO do zero, seguindo rigorosamente o esboço fornecido e resolvendo *todos* os problemas listados no feedback crítico.
Preste atenção especial a gramática, ortografia, completude e tom amigável-profissional.
"""
        user_content = f"""
TÓPICO: {topic}
ESBOÇO:
{outline}

FEEDBACK DE REVISÃO:
{feedback}

REESCREVA O ARTIGO COMPLETO.
"""
    else:
        print("--- 2a. EXECUTANDO REDAÇÃO: Gerando Rascunho Inicial ---")
        system_content = "Você é um redator profissional. Redija o conteúdo COMPLETO do artigo com base no esboço, mantendo tom informativo e amigável."
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

def review_content(state: ContentAgentState):
    print(f"--- 3. EXECUTANDO REVISÃO (Revisão {state.get('revision_count',0)}) ---")
    content = state["content"]
    
    review_prompt = f"""
Avalie o rascunho do artigo segundo:
1. Completeness (Todas as seções do esboço presentes?)
2. Gramática e ortografia
3. Tom adequado

Se tudo estiver ok, responda "OK". Se houver problemas, responda "REVISÃO:" seguido de lista numerada de correções necessárias.

Primeiros 1000 caracteres do artigo:\n{content[:1000]}...
"""
    
    review_chain = ChatPromptTemplate.from_template(review_prompt) | llm
    full_review_result = review_chain.invoke({}).content.strip()

    result_keyword = full_review_result.split(':')[0].strip().upper()

    # Limite máximo de revisões = 5
    MAX_REVISIONS = 5
    if state.get("revision_count",0) >= MAX_REVISIONS:
        print(f"⚠️ Limite de revisões ({MAX_REVISIONS}) atingido. Forçando publicação.")
        return {"revision_needed": False, "review_feedback": "OK"}

    if "REVISÃO" in result_keyword:
        print("!! REVISÃO NECESSÁRIA: Voltando para Redação.")
        return {"revision_needed": True, "review_feedback": full_review_result}
    else:
        print("✅ APROVADO: Conteúdo pronto para publicação.")
        return {"revision_needed": False, "review_feedback": "OK"}

# ====================================================================
# III. LÓGICA DO GRAFO
# ====================================================================

def should_continue(state: ContentAgentState):
    if state["revision_needed"] and state.get("revision_count",0) < 5:
        return "re_draft"
    else:
        return "publish"

def create_agent_graph():
    workflow = StateGraph(ContentAgentState)

    workflow.add_node("plan", plan_content)
    workflow.add_node("draft", draft_content)
    workflow.add_node("review", review_content)

    workflow.set_entry_point("plan")
    workflow.add_edge("plan", "draft")
    workflow.add_edge("draft", "review")
    workflow.add_conditional_edges(
        "review",
        should_continue,
        {"re_draft": "draft", "publish": END}
    )

    # Mantém recursion_limit padrão (25)
    app = workflow.compile()
    return app

# ====================================================================
# IV. EXECUÇÃO
# ====================================================================

if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "Content-Generator-Agent") 
    
    if not LANGCHAIN_API_KEY or not GROQ_API_KEY:
        print("ERRO: Chaves LANGCHAIN_API_KEY e GROQ_API_KEY são necessárias.")
    else:
        app = create_agent_graph()
        initial_topic = input("Insira o tópico para o artigo (ou deixe vazio para sair): ").strip()
        if not initial_topic:
            print("Nenhum tópico fornecido. A sair.")
            exit()

        print(f"\n>>>> INICIANDO AGENTE PARA O TÓPICO: {initial_topic} <<<<\n")
        final_state = app.invoke({"topic": initial_topic})
        final_content = final_state["content"]

        print("\n\n#####################################################")
        print("########## AGENTE CONCLUÍDO. CONTEÚDO FINAL ###########")
        print("#####################################################")
        print(final_content)
