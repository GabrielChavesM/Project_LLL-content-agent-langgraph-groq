import os
import json
from typing import TypedDict
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from langgraph.graph import StateGraph, END

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

class ContentAgentState(TypedDict):
    topic: str
    outline: str
    content: str
    revision_needed: bool
    review_feedback: str
    revision_count: int


# ====================================================================
# 1. PLANNING NODE
# ====================================================================
def plan_content(state: ContentAgentState):
    print("--- 1. PLANEAMENTO: Criando Esboço ---")

    topic = state["topic"]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Você é um planeador de conteúdo especialista.
Crie um esboço detalhado em Markdown usando ## para seções.
O esboço deve ser completo, lógico e cobrindo todo o tema.
"""),
        HumanMessage(content=f"Tópico: {topic}")
    ])

    outline = (prompt | llm).invoke({}).content

    return {
        "outline": outline,
        "content": "",
        "review_feedback": "",
        "revision_count": 0,
        "revision_needed": False
    }


# ====================================================================
# 2. DRAFT / REDRAFT NODE
# ====================================================================
def draft_content(state: ContentAgentState):
    topic = state["topic"]
    outline = state["outline"]
    feedback = state.get("review_feedback", "")
    revision_count = state.get("revision_count", 0) + 1

    is_redraft = bool(feedback and feedback.strip() != "OK")

    if is_redraft:
        print(f"--- 2b. REDAÇÃO (REVISÃO {revision_count}) ---")

        system_content = """
Você é um redator profissional.

- Reescreva o artigo COMPLETO do zero
- Siga rigorosamente o esboço
- Corrija todos os problemas do feedback
- Melhore clareza, gramática e estrutura
- NÃO inclua comentários
"""

        user_content = f"""
TÓPICO:
{topic}

ESBOÇO:
{outline}

FEEDBACK:
{feedback}
"""
    else:
        print("--- 2a. REDAÇÃO INICIAL ---")

        system_content = """
Você é um redator profissional.
Escreva um artigo completo, bem estruturado,
com tom informativo e amigável-profissional.
"""

        user_content = f"""
TÓPICO:
{topic}

ESBOÇO:
{outline}
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ])

    content = (prompt | llm).invoke({}).content

    return {
        "content": content,
        "revision_needed": is_redraft,
        "review_feedback": feedback,
        "revision_count": revision_count
    }


# ====================================================================
# 3. REVIEW NODE
# ====================================================================
def review_content(state: ContentAgentState):
    print(f"--- 3. REVISÃO (Tentativa {state.get('revision_count', 0)}) ---")

    content = state["content"]

    system_prompt = """
Você é um revisor editorial rigoroso.

Responda APENAS em JSON válido.

CRITÉRIOS:
- completeness
- grammar_spelling
- tone
- clarity
- structure

REGRAS:
- Não seja vago
- Sempre dê sugestões concretas
- Se estiver perfeito, retorne {"status": "OK"}
"""

    user_prompt = f"""
Avalie o artigo abaixo:

{content}

Retorne JSON no formato:

{{
  "status": "REVISAR",
  "scores": {{
    "completeness": "OK|REVISAR",
    "grammar_spelling": "OK|REVISAR",
    "tone": "OK|REVISAR",
    "clarity": "OK|REVISAR",
    "structure": "OK|REVISAR"
  }},
  "issues": [
    {{
      "type": "grammar|tone|structure|completeness|clarity",
      "description": "descrição objetiva",
      "suggestion": "como corrigir"
    }}
  ]
}}
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])

    result = (prompt | llm).invoke({}).content.strip()

    try:
        parsed = json.loads(result)
    except json.JSONDecodeError:
        print("⚠️ JSON inválido -> forçando revisão")

        return {
            "revision_needed": True,
            "review_feedback": json.dumps({
                "status": "REVISAR",
                "issues": [{
                    "type": "system",
                    "description": "JSON inválido do modelo",
                    "suggestion": "Corrigir formato de saída"
                }]
            })
        }

    MAX_REVISIONS = 5

    if state.get("revision_count", 0) >= MAX_REVISIONS:
        print("Limite de revisões atingido")
        return {"revision_needed": False, "review_feedback": "OK"}

    if parsed.get("status") == "OK":
        print("APROVADO")
        return {
            "revision_needed": False,
            "review_feedback": "OK"
        }

    print("🔁 REVISÃO NECESSÁRIA")

    return {
        "revision_needed": True,
        "review_feedback": json.dumps(parsed, ensure_ascii=False)
    }


# ====================================================================
# 4. ROUTING
# ====================================================================
def should_continue(state: ContentAgentState):
    if state["revision_needed"] and state.get("revision_count", 0) < 5:
        return "re_draft"
    return "publish"


# ====================================================================
# 5. GRAPH
# ====================================================================
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
        {
            "re_draft": "draft",
            "publish": END
        }
    )

    return workflow.compile()


# ====================================================================
# 6. EXECUTION
# ====================================================================
if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT", "Content-Generator-Agent")

    if not GROQ_API_KEY or not LANGCHAIN_API_KEY:
        print("ERRO: faltam variáveis de ambiente")
        exit()

    app = create_agent_graph()

    topic = input("Insira o tópico: ").strip()

    if not topic:
        print("Nenhum tópico fornecido.")
        exit()

    print(f"\nINICIANDO: {topic}\n")

    final_state = app.invoke({
        "topic": topic,
        "outline": "",
        "content": "",
        "revision_needed": False,
        "review_feedback": "",
        "revision_count": 0
    })

    print("\n" + "#" * 60)
    print("CONTEÚDO FINAL")
    print("#" * 60)
    print(final_state["content"])