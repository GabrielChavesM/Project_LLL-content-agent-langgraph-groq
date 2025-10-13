import os
import operator
import uuid
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import requests
import re

# --- LangChain / Hugging Face / Chroma Imports ---
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# LangChain embeddings
from langchain_huggingface import HuggingFaceEmbeddings

# ChromaDB
from chromadb import PersistentClient
from chromadb.utils import embedding_functions

# --- LangGraph Imports ---
from langgraph.graph import StateGraph, END

# --- CrewAI Imports ---
from crewai import Task
from crewai_tools import ScrapeWebsiteTool, SerperDevTool
from bs4 import BeautifulSoup

# ====================================================================
# 0. CARREGAR VARIÁVEIS DE AMBIENTE
# ====================================================================

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Variáveis para ChromaDB / persistência
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "content_collection")

# ====================================================================
# I. CONFIGURAÇÃO DO LLM (LangChain)
# ====================================================================

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7,
    groq_api_key=GROQ_API_KEY
)

# ====================================================================
# I.b CONFIGURAÇÃO DO EMBEDDINGS (HuggingFace) e CHROMADB
# ====================================================================

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

try:
    chroma_client = PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL_NAME
    )
    chroma_collection = chroma_client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        embedding_function=chroma_ef
    )
    print(f"✅ ChromaDB inicializada em: {CHROMA_PERSIST_DIR}, coleção: {CHROMA_COLLECTION_NAME}")
except Exception as e:
    chroma_client = None
    chroma_collection = None
    print(f"⚠️ Falha ao inicializar ChromaDB: {e}. O RAG continuará, mas sem persistência local.")

# ====================================================================
# II. DEFINIÇÃO DO ESTADO DO GRAFO
# ====================================================================

class ContentAgentState(TypedDict):
    topic: str
    research_data: Annotated[str, operator.add]
    outline: Annotated[str, operator.add]
    content: Annotated[str, operator.add]
    revision_needed: bool
    review_feedback: str
    revision_count: int

# ====================================================================
# Utilitários para Chroma / RAG
# ====================================================================

def index_texts_to_chroma(texts, metadatas=None, ids=None):
    global chroma_collection, embed_model
    if chroma_collection is None:
        print("⚠️ Chroma não disponível — salto ao armazenamento em vetor.")
        return None

    if not ids:
        ids = [str(uuid.uuid4()) for _ in texts]
    if not metadatas:
        metadatas = [{} for _ in texts]

    try:
        embeddings = embed_model.embed_documents(texts)
        chroma_collection.add(
            documents=texts,
            metadatas=metadatas,
            ids=ids,
            embeddings=embeddings
        )
        print(f"✅ Indexados {len(texts)} documentos em Chroma (coleção: {CHROMA_COLLECTION_NAME}) — persistência automática habilitada.")
        return ids

    except Exception as e:
        print(f"⚠️ Erro ao indexar em Chroma: {e}")
        return None

def retrieve_from_chroma(query, n_results=4):
    global chroma_collection
    if chroma_collection is None:
        print("⚠️ Chroma não inicializada — nenhuma recuperação disponível.")
        return []

    try:
        res = chroma_collection.query(query_texts=[query], n_results=n_results, include=["documents", "metadatas"]) or {}
        docs = []
        docs_list = res.get("documents", [[]])[0]
        metas_list = res.get("metadatas", [[]])[0]
        for d, m in zip(docs_list, metas_list):
            docs.append({"text": d, "metadata": m})
        print(f"🔎 Recuperados {len(docs)} documentos relevantes de Chroma para a query")
        return docs
    except Exception as e:
        print(f"⚠️ Erro na recuperação de Chroma: {e}")
        return []

# ====================================================================
# SUMARIZAÇÃO E ARMAZENAMENTO DE CONTEXTO
# ====================================================================

def summarize_and_store(stage: str, text: str, topic: str):
    if not text or len(text.strip()) < 50:
        return None

    try:
        summary_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um resumidor profissional."),
            HumanMessage(content=(
                f"Resuma o conteúdo abaixo em no máximo 200 palavras, "
                f"destacando apenas os pontos-chave, nomes e conceitos relevantes.\n\n"
                f"Texto:\n{text}"
            ))
        ])
        chain = summary_prompt | llm
        summary = chain.invoke({}).content.strip()

        doc_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{topic}-{stage}-{uuid.uuid4()}"))
        metadata = {"stage": stage, "topic": topic}
        index_texts_to_chroma([summary], metadatas=[metadata], ids=[doc_id])
        print(f"🧠 Resumo da etapa '{stage}' armazenado no ChromaDB.")
        return summary

    except Exception as e:
        print(f"⚠️ Erro ao resumir/armazenar contexto ({stage}): {e}")
        return None

# ====================================================================
# III. NÓ DE PESQUISA NA WEB
# ====================================================================

def web_research(state: ContentAgentState):
    print("--- 0. EXECUTANDO PESQUISA NA WEB ---")
    topic = state["topic"]
    urls = []

    if SERPER_API_KEY:
        try:
            search_tool = SerperDevTool()
            results = search_tool._run(query=topic)
            for r in results.get("organic", [])[:5]:
                link = r.get("link")
                if link and not link.startswith("http"):
                    link = "https://" + link.lstrip("/")
                if link:
                    urls.append(link)
        except Exception as e:
            print(f"⚠️ Erro ao usar SerperDevTool ({e}).")

    research_texts = []
    metadatas = []
    headers = {"User-Agent": "Mozilla/5.0"}

    for url in urls:
        for attempt in range(3):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    paragraphs = soup.find_all("p")
                    text = "\n".join(p.get_text() for p in paragraphs if p.get_text())
                    if text and len(text) > 100:
                        snippet = text[:3000]
                        research_texts.append(snippet)
                        metadatas.append({"source": url})
                    break
            except Exception as e:
                print(f"⚠️ Falha ao extrair {url}: {e}")

    combined_research = "\n\n---\n\n".join(research_texts) if research_texts else "Nenhuma informação relevante encontrada."
    if research_texts:
        try:
            ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, m.get('source', str(i)))) for i, m in enumerate(metadatas)]
            index_texts_to_chroma(research_texts, metadatas=metadatas, ids=ids)
        except Exception as e:
            print(f"⚠️ Erro ao indexar/pesquisar Chroma: {e}")

    summarize_and_store("web_research", combined_research, topic)
    return {"research_data": combined_research}

# ====================================================================
# V.b VALIDAÇÃO E LIMPEZA DE ESTRUTURA DO TEXTO
# ====================================================================

def enforce_structure(text: str) -> str:
    required_sections = ["Introdução", "História", "Conceitos", "Conclusão", "Referências"]
    missing = [s for s in required_sections if s not in text]

    if missing:
        print(f"⚠️ Faltam seções: {missing}. Solicitando reescrita estruturada...")
        repair_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="Você é um reestruturador de artigos."),
            HumanMessage(content=(
                f"O texto abaixo está faltando seções obrigatórias ({', '.join(missing)}). "
                f"Reescreva o artigo mantendo o conteúdo, mas garantindo que todas as seções padrão apareçam no formato Markdown. "
                f"Retorne apenas UMA versão final estruturada (sem repetir o texto original).\n\n"
                f"Texto original:\n{text}"
            ))
        ])
        chain = repair_prompt | llm
        fixed = chain.invoke({}).content
        text = fixed

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'#+\s*', lambda m: m.group(0).strip() + ' ', text)
    return text.strip()

def remove_duplicate_sections(text: str) -> str:
    sections = ["Introdução", "História / Contexto", "Conceitos Fundamentais",
                "Comparações / Diferenciações", "Exemplos Práticos",
                "Tópicos Avançados / Recomendações", "Conclusão", "Referências"]

    section_positions = {}
    for sec in sections:
        pos_list = [m.start() for m in re.finditer(fr"##\s*{re.escape(sec)}", text)]
        if pos_list:
            section_positions[sec] = pos_list[-1]

    sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
    cleaned_text = ""
    for i, (sec, pos) in enumerate(sorted_sections):
        next_pos = sorted_sections[i+1][1] if i+1 < len(sorted_sections) else len(text)
        cleaned_text += text[pos:next_pos].strip() + "\n\n"
    return cleaned_text.strip()

def clean_output(text: str) -> str:
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'#+\s*', lambda m: m.group(0).strip() + ' ', text)
    return text.strip()

# ====================================================================
# IV. PLANEAMENTO DE CONTEÚDO
# ====================================================================

def plan_content(state: ContentAgentState):
    print("--- 1. EXECUTANDO PLANEAMENTO ---")
    topic = state["topic"]
    research_data = state.get("research_data", "")

    structure = """
Siga rigorosamente o formato abaixo (em Markdown):

## Título
## Introdução
## História / Contexto
## Conceitos Fundamentais
## Comparações / Diferenciações
## Exemplos Práticos
## Tópicos Avançados / Recomendações
## Conclusão
## Referências
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"Você é um planejador de conteúdo. Crie um esboço detalhado seguindo esta estrutura:\n{structure}"),
        HumanMessage(content=f"Tópico: {topic}\n\nInformações coletadas:\n{research_data}")
    ])

    chain = prompt | llm
    outline_result = chain.invoke({"topic": topic}).content
    summarize_and_store("plan_content", outline_result, topic)
    return {"outline": outline_result, "content": "", "review_feedback": "", "revision_count": 0}

# ====================================================================
# V. REDAÇÃO DO CONTEÚDO
# ====================================================================

def draft_content(state: ContentAgentState):
    topic = state["topic"]
    outline = state["outline"]
    feedback = state.get("review_feedback", "")
    revision_count = state.get("revision_count", 0) + 1
    is_redraft = bool(feedback) and feedback.strip().upper() not in ["OK", ""]

    structure_instructions = """
<FORMATO_ESPERADO>
## Título
## Introdução
## História / Contexto
## Conceitos Fundamentais
## Comparações / Diferenciações
## Exemplos Práticos
## Tópicos Avançados / Recomendações
## Conclusão
## Referências
</FORMATO_ESPERADO>
Respeite exatamente este formato.
"""

    retrieved_context = ""
    try:
        retrieved = retrieve_from_chroma(topic, n_results=4)
        if retrieved:
            pieces = [f"Fonte: {d['metadata'].get('source','-')}\n{d['text']}" for d in retrieved]
            retrieved_context = "\n\n---\n\n".join(pieces)
            print("📥 Contexto recuperado do RAG incorporado ao prompt.")
    except Exception as e:
        print(f"⚠️ Falha ao recuperar contexto RAG: {e}")

    if is_redraft:
        print(f"--- 2b. REESCREVENDO RASCUNHO (REVISÃO {revision_count}) ---")
        system_content = "Você é um redator profissional. Reescreva o artigo do zero resolvendo todos os pontos do feedback."
        user_content = f"Tópico: {topic}\n\nEsboço:\n{outline}\n\nFeedback:\n{feedback}\n{structure_instructions}\n\nContexto adicional:\n{retrieved_context}"
    else:
        print("--- 2a. GERANDO RASCUNHO INICIAL ---")
        system_content = "Você é um redator profissional. Redija o artigo completo seguindo o esboço e formato solicitado."
        user_content = f"Tópico: {topic}\n\nEsboço:\n{outline}\n{structure_instructions}\n\nContexto adicional:\n{retrieved_context}"

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content=user_content)
    ])

    chain = prompt | llm
    content_result = chain.invoke({}).content

    # ✅ Validação e limpeza automática
    content_result = enforce_structure(content_result)
    content_result = remove_duplicate_sections(content_result)
    content_result = clean_output(content_result)
    summarize_and_store("draft_content", content_result, topic)

    return {"content": content_result, "revision_needed": is_redraft, "review_feedback": feedback, "revision_count": revision_count}

# ====================================================================
# VI. REVISÃO COM CREWAI (GROQ)
# ====================================================================

class GroqAgentWrapper:
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
    print("--- 3. EXECUTANDO REVISÃO COM CREWAI ---")
    content = state["content"]
    topic = state["topic"]
    groq_agent = GroqAgentWrapper(llm)

    for iteration in range(3):
        print(f"🔄 Iteração {iteration+1} da revisão")
        edit_task = Task(
            description=f"Reescreva o artigo abaixo aplicando correções necessárias, mas NÃO adicione comentários, listas de melhorias ou revisões técnicas. Apenas mantenha o conteúdo no formato Markdown solicitado.\n\n{content}",
            expected_output="Artigo revisado sem comentários.",
            agent=None
        )
        final_result = groq_agent.execute_task(edit_task)

        # ✅ Estrutura correta, remover duplicações e limpar
        final_result = enforce_structure(final_result)
        final_result = remove_duplicate_sections(final_result)
        final_result = clean_output(final_result)

        if "MELHORAR" not in final_result.upper():
            print("✅ Revisão finalizada.")
            summarize_and_store("crew_review", final_result, topic)
            return {"revision_needed": False, "review_feedback": "OK", "content": final_result.strip()}

        content = final_result

    print("⚠️ Revisão necessária após 3 iterações.")
    summarize_and_store("crew_review", content, topic)
    return {"revision_needed": True, "review_feedback": final_result, "content": final_result}

# ====================================================================
# VII. LÓGICA DO GRAFO
# ====================================================================

def should_continue(state: ContentAgentState):
    if state["revision_needed"] and state.get("revision_count", 0) < 5:
        return "re_draft"
    return "publish"

def create_agent_graph():
    workflow = StateGraph(ContentAgentState)
    workflow.add_node("web_research", web_research)
    workflow.add_node("plan", plan_content)
    workflow.add_node("draft", draft_content)
    workflow.add_node("crew_review", crew_review_content)
    workflow.set_entry_point("web_research")
    workflow.add_edge("web_research", "plan")
    workflow.add_edge("plan", "draft")
    workflow.add_edge("draft", "crew_review")
    workflow.add_conditional_edges("crew_review", should_continue, {"re_draft": "draft", "publish": END})
    return workflow.compile()

# ====================================================================
# VIII. EXECUÇÃO FINAL
# ====================================================================

if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGCHAIN_API_KEY or "")
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

        # Garantir estrutura correta, remover duplicações e limpar
        final_content = enforce_structure(final_content)
        final_content = remove_duplicate_sections(final_content)
        final_content = clean_output(final_content)

        print("\n" + "#" * 55)
        print("#" * 10 + " CONTEÚDO FINAL GERADO PELO AGENTE " + "#" * 10)
        print("#" * 55 + "\n")
        print(final_content)
        print("\n" + "#" * 55 + "\n")
