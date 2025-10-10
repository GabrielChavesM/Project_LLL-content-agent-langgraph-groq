import streamlit as st
import os
import operator
from typing import TypedDict, Annotated
from dotenv import load_dotenv

# --- LangChain Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# ==========================================================
# I. CONFIGURAÇÃO DE AMBIENTE E LLM
# ==========================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

# Configuração do modelo Groq via LangChain
llm = ChatGroq(
    model_name="llama-3.1-8b-instant",  # Modelo recomendado após descontinuação do Gemma
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

# ==========================================================
# II. DEFINIÇÃO DO ESTADO DO GRAFO (Contexto compartilhado)
# ==========================================================

class ContentAgentState(TypedDict):
    topic: str
    outline: Annotated[str, operator.add]
    content: Annotated[str, operator.add]
    revision_needed: bool
    review_feedback: str
    revision_count: int  # Contador de revisões

# ==========================================================
# III. DEFINIÇÃO DOS NÓS DO GRAFO
# ==========================================================

# 1️⃣ PLANEJAMENTO
def plan_content(state: ContentAgentState):
    topic = state["topic"]
    st.info("🧭 Gerando esboço do artigo...")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=(
            "Você é um planejador de conteúdo experiente. "
            "Crie um esboço detalhado (em Markdown com ## para seções) "
            "para um artigo sobre o tópico fornecido. O esboço deve ser lógico e completo."
        )),
        HumanMessage(content=f"Tópico: {topic}")
    ])

    chain = prompt | llm
    outline_result = chain.invoke({"topic": topic}).content

    return {"outline": outline_result, "content": "", "review_feedback": "", "revision_count": 0}


# 2️⃣ REDAÇÃO
def draft_content(state: ContentAgentState):
    topic = state["topic"]
    outline = state["outline"]
    feedback = state.get("review_feedback", "")
    revision_count = state.get("revision_count", 0) + 1
    is_redraft = bool(feedback) and feedback.strip().upper() not in ["OK", ""]

    if is_redraft:
        st.info(f"✏️ Reescrevendo artigo (revisão {revision_count})...")
        system_content = (
            "Você é um redator profissional altamente detalhista. "
            "Reescreva o artigo completo do zero, seguindo o esboço fornecido "
            "e corrigindo todos os problemas apontados no feedback."
        )
        user_content = f"TÓPICO: {topic}\nESBOÇO:\n{outline}\n\nFEEDBACK:\n{feedback}"
    else:
        st.info("📝 Gerando rascunho inicial...")
        system_content = (
            "Você é um redator profissional. "
            "Redija o conteúdo completo do artigo com base no esboço, "
            "mantendo tom informativo, amigável e profissional."
        )
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


# 3️⃣ REVISÃO
def review_content(state: ContentAgentState):
    content = state["content"]
    revision_num = state.get("revision_count", 0)

    st.info(f"🔎 Avaliando conteúdo (revisão {revision_num})...")

    review_prompt = f"""
Você é um editor de conteúdo rigoroso.
Avalie o artigo conforme os critérios:

1. Completeness (todas as seções prometidas estão presentes?)
2. Gramática e ortografia impecáveis?
3. Tom adequado (informativo, amigável e profissional)?

Se tudo estiver bom, responda "OK".
Caso contrário, responda com "REVISÃO:" seguido de uma lista numerada dos problemas a corrigir.

Texto (primeiros 1000 caracteres):
---
{content[:1000]}...
"""

    review_chain = ChatPromptTemplate.from_template(review_prompt) | llm
    review_result = review_chain.invoke({}).content.strip()
    result_keyword = review_result.split(':')[0].strip().upper()

    MAX_REVISIONS = 5
    if revision_num >= MAX_REVISIONS:
        st.warning("⚠️ Limite máximo de revisões atingido. Publicando versão atual.")
        return {"revision_needed": False, "review_feedback": "OK"}

    if "REVISÃO" in result_keyword:
        return {"revision_needed": True, "review_feedback": review_result}
    else:
        st.success("✅ Conteúdo aprovado!")
        return {"revision_needed": False, "review_feedback": "OK"}


# ==========================================================
# IV. LÓGICA DE ROTEAMENTO DO GRAFO
# ==========================================================

def should_continue(state: ContentAgentState):
    if state["revision_needed"] and state.get("revision_count", 0) < 5:
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

    return workflow.compile()


# ==========================================================
# V. INTERFACE STREAMLIT
# ==========================================================

st.set_page_config(page_title="Content Generator Agent", layout="wide")
st.title("🧠 Gerador de Artigos com LangChain + LangGraph")

if not LANGCHAIN_API_KEY or not GROQ_API_KEY:
    st.error("⚠️ As chaves `LANGCHAIN_API_KEY` e `GROQ_API_KEY` devem estar definidas no .env")
else:
    topic_input = st.text_input("Digite o tópico do artigo:")
    generate_btn = st.button("🚀 Gerar Artigo")

    if generate_btn and topic_input.strip():
        with st.spinner("Executando agente de conteúdo..."):
            app = create_agent_graph()
            final_state = app.invoke({"topic": topic_input.strip()})
            final_content = final_state["content"]
            revision_count = final_state.get("revision_count", 0)
            outline = final_state.get("outline", "")
            feedback = final_state.get("review_feedback", "")

        st.success(f"✅ Artigo concluído após {revision_count} revisão(ões).")

        with st.expander("📑 Esboço Gerado (Outline)"):
            st.markdown(outline)

        with st.expander("🧾 Feedback da Revisão Final"):
            st.text(feedback)

        st.subheader("📝 Conteúdo Final")
        st.text_area("Artigo Completo", value=final_content, height=500)

