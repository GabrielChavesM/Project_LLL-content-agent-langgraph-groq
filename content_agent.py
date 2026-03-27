import os
import json
import re
import time
import requests
import httpx
from pathlib import Path
from typing import TypedDict, List
from dotenv import load_dotenv

from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

load_dotenv()
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
SERPER_API_KEY    = os.getenv("SERPER_API_KEY")

llm_fast = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=OPENAI_API_KEY,
)

llm_writer = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    api_key=OPENAI_API_KEY,
)

SKIP_DOMAINS = {
    "youtube.com", "youtu.be",
    "instagram.com", "twitter.com", "x.com",
    "facebook.com", "tiktok.com",
    "reddit.com",
}

class ContentAgentState(TypedDict):
    topic:             str
    outline:           str
    content:           str
    revision_needed:   bool
    review_feedback:   str
    revision_count:    int
    research_data:     str
    raw_sources:       str
    verified_entities: str
    fact_errors:       str

# ====================================================================
# UTILS
# ====================================================================
def is_scrapable(url: str) -> bool:
    return not any(domain in url for domain in SKIP_DOMAINS)


def scrape_article(url: str, timeout: int = 12, max_chars: int = 6000) -> str:
    """Extrai texto editorial de um URL. Ignora domínios de vídeo/social."""
    if not is_scrapable(url):
        return f"[Ignorado — domínio de vídeo/social: {url}]"
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            ),
            "Accept-Language": "pt-BR,pt;q=0.9,en;q=0.8",
        }
        r = httpx.get(url, timeout=timeout, follow_redirects=True, headers=headers)
        r.raise_for_status()

        soup = BeautifulSoup(r.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "aside",
                         "header", "form", "noscript"]):
            tag.decompose()

        main = soup.find("article") or soup.find("main") or soup
        text  = main.get_text(separator="\n", strip=True)
        lines = [ln.strip() for ln in text.splitlines() if len(ln.strip()) > 50]
        return "\n".join(lines)[:max_chars]
    except Exception as e:
        return f"[Erro ao extrair {url}: {e}]"


# ====================================================================
# 0. RESEARCH NODE
# ====================================================================
def research_topic(state: ContentAgentState) -> dict:
    print("--- 0. RESEARCH: Pesquisando e extraindo conteúdo real ---")
    topic = state["topic"]

    # 0a. Busca no Serper
    try:
        resp = requests.post(
            "https://google.serper.dev/search",
            json={"q": topic, "num": 20, "hl": "pt"},
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            timeout=15,
        )
        resp.raise_for_status()
        organic = resp.json().get("organic", [])
    except Exception as e:
        print(f"   ⚠️ Erro na pesquisa: {e}")
        organic = []

    if not organic:
        return {"research_data": "Sem resultados.", "raw_sources": "", "verified_entities": "{}"}

    # 0b. Scraping - até 8 fontes scrapáveis, ignora vídeo/social
    scrapable   = [item for item in organic if is_scrapable(item.get("link", ""))]
    top_results = scrapable[:8]
    scraped_parts: List[str] = []
    fonte_idx = 1

    for item in top_results:
        url     = item.get("link", "")
        title   = item.get("title", f"Artigo {fonte_idx}")
        snippet = item.get("snippet", "")

        print(f"   Scraping {fonte_idx}/{len(top_results)}: {url[:90]}")
        body = scrape_article(url)

        if body.startswith("[Erro") or body.startswith("[Ignorado"):
            print(f"      ↳ {body[:80]}")
            continue

        scraped_parts.append(
            f"## FONTE {fonte_idx}: {title}\n"
            f"URL: {url}\n"
            f"Snippet: {snippet}\n\n"
            f"{body}"
        )
        fonte_idx += 1
        time.sleep(0.3)

    # Fallback: snippets do Serper se nenhuma fonte foi scrapeada
    if not scraped_parts:
        print("   ⚠️ Sem fontes scrapáveis. Usando snippets do Serper.")
        scraped_parts = [
            f"## FONTE {i+1}: {item.get('title')}\nURL: {item.get('link')}\n{item.get('snippet')}"
            for i, item in enumerate(organic[:10])
        ]

    raw_sources = ("\n\n" + "=" * 60 + "\n\n").join(scraped_parts)

    # 0c. Sumarização rigorosa
    summary_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Você é um assistente de pesquisa rigoroso.

Crie um resumo estruturado em Markdown APENAS com informações EXPLÍCITAS nas fontes.

Regras:
- Para cada facto, indique a fonte (ex: "Fonte 2").
- Inclua: nomes completos, datas, resultados, valores, cargos, transferências.
- NÃO invente, complete ou assuma informação ausente.
- Se um facto aparecer em múltiplas fontes, mencione-as.
"""),
        HumanMessage(content=f"Tópico: {topic}\n\nFontes:\n{raw_sources[:12000]}"),
    ])

    try:
        research_summary = (summary_prompt | llm_fast).invoke({}).content
    except Exception as e:
        research_summary = f"Erro na sumarização: {e}"

    time.sleep(0.5)
    return {"research_data": research_summary, "raw_sources": raw_sources}

# ====================================================================
# 1. EXTRACT ENTITIES NODE
# ====================================================================
def extract_entities(state: ContentAgentState) -> dict:
    print("--- 1. ENTIDADES: Extraindo factos verificados ---")
    raw_sources = state.get("raw_sources", "")
    topic       = state["topic"]

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Você é um extractor de entidades rigoroso.

Analise as fontes e extraia APENAS entidades mencionadas explicitamente.
Retorne SOMENTE JSON válido, sem texto antes ou depois.

Formato:
{
  "pessoas": [
    {"nome": "Nome Completo", "papel": "cargo/função", "detalhes": "ex: contrato até X", "fonte": "Fonte N"}
  ],
  "jogadores_confirmados": [
    {"nome": "Nome Completo", "posicao": "...", "situacao": "contratado/emprestado/saiu/fica", "destino_origem": "clube", "fonte": "Fonte N"}
  ],
  "competicoes": ["Campeonato X", "Copa Y"],
  "valores_financeiros": [
    {"descricao": "...", "valor": "...", "fonte": "Fonte N"}
  ],
  "datas_confirmadas": [
    {"evento": "...", "data": "...", "fonte": "Fonte N"}
  ],
  "estatisticas_equipe": [
    {"metrica": "...", "valor": "...", "fonte": "Fonte N"}
  ],
  "outros_factos": [
    {"facto": "descrição completa e específica", "fonte": "Fonte N"}
  ]
}

NUNCA invente. Se não estiver na fonte, não inclua.
"""),
        HumanMessage(content=f"Tópico: {topic}\n\nFontes:\n{raw_sources[:10000]}"),
    ])

    result = (prompt | llm_fast).invoke({}).content.strip()
    time.sleep(0.5)

    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    if json_match:
        try:
            parsed            = json.loads(json_match.group())
            verified_entities = json.dumps(parsed, ensure_ascii=False, indent=2)
            n_j = len(parsed.get("jogadores_confirmados", []))
            n_p = len(parsed.get("pessoas", []))
            print(f"   ✅ {n_j} jogadores, {n_p} pessoas extraídas.")
        except json.JSONDecodeError:
            verified_entities = "{}"
            print("   ⚠️ JSON inválido. Usando vazio.")
    else:
        verified_entities = "{}"
        print("   ⚠️ Sem JSON encontrado.")

    return {"verified_entities": verified_entities}

# ====================================================================
# 2. PLANNING NODE
# ====================================================================
def plan_content(state: ContentAgentState) -> dict:
    print("--- 2. PLANEAMENTO: Criando esboço detalhado ---")
    topic    = state["topic"]
    research = state.get("research_data", "")
    entities = state.get("verified_entities", "{}")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Você é um editor-chefe de jornalismo desportivo.

Crie um esboço detalhado para um artigo COMPLETO, APROFUNDADO e JORNALÍSTICO.
Use `##` para secções principais e `###` para subseções.

O esboço deve:
- Ter introdução com contexto, relevância e gancho narrativo
- Ter corpo com múltiplas secções temáticas ricas e bem distintas
- Para cada secção, listar os PONTOS ESPECÍFICOS que o redactor deve abordar
- Terminar com perspectivas/conclusão e lista de fontes
- Basear-se APENAS nos factos verificados disponíveis
- Omitir secções sem dados

O artigo final deve ter mínimo 1200 palavras em prosa corrida.
"""),
        HumanMessage(content=f"""
Tópico: {topic}

Resumo de pesquisa:
{research}

Entidades verificadas (JSON):
{entities}

Crie o esboço.
"""),
    ])

    outline = (prompt | llm_writer).invoke({}).content
    time.sleep(0.5)

    return {
        "outline":         outline,
        "content":         "",
        "review_feedback": "",
        "revision_count":  0,
        "revision_needed": False,
        "fact_errors":     "",
    }

# ====================================================================
# 3. DRAFT / REDRAFT NODE
# ====================================================================
ANTI_HALLUCINATION = """
REGRAS ABSOLUTAS (violá-las invalida o artigo):
1. Use APENAS nomes completos de pessoas e jogadores presentes nas ENTIDADES VERIFICADAS.
2. Se o nome completo de um jogador não estiver nas entidades → NÃO o mencione.
3. NUNCA invente valores financeiros (orçamentos, salários, cláusulas).
4. NUNCA atribua historial profissional a ninguém sem confirmação nas fontes.
5. Para transferências sem nome confirmado, mencione o papel ("um dos avançados") sem inventar.
6. Informação ausente = omita. Não escreva "informação não disponível" repetidamente.
"""

WRITING_STYLE = """
ESTILO JORNALÍSTICO:
- Escreva em PROSA CORRIDA, como um artigo de revista desportiva de qualidade.
- Use parágrafos de 4-6 linhas. NUNCA use bullet lists como corpo do artigo.
- Conecte factos com contexto, análise e narrativa.
- Varie o ritmo: alterne frases curtas e longas para criar ritmo.
- Cite nomes e dados concretos sempre que disponíveis nas entidades.
- Tom: jornalístico, informativo, envolvente e profissional.
- Use transições entre parágrafos para criar fluidez.
"""

def draft_content(state: ContentAgentState) -> dict:
    topic          = state["topic"]
    outline        = state["outline"]
    research       = state.get("research_data", "")
    entities       = state.get("verified_entities", "{}")
    feedback       = state.get("review_feedback", "")
    fact_errors    = state.get("fact_errors", "")
    revision_count = state.get("revision_count", 0) + 1

    is_redraft = bool(
        (feedback and feedback.strip() not in ("", "OK")) or
        (fact_errors and fact_errors.strip() not in ("", "{}"))
    )

    if is_redraft:
        print(f"--- 3b. REDAÇÃO (REVISÃO {revision_count}) ---")
        system_content = f"""
Você é um jornalista especialista em futebol.

{ANTI_HALLUCINATION}
{WRITING_STYLE}

Reescreva o artigo COMPLETO do zero.
- Corrija TODOS os problemas do feedback de estilo.
- Corrija TODOS os erros factuais do fact-check.
- Mínimo de 1200 palavras em prosa.
- NÃO inclua meta-comentários, notas ou avisos no corpo.
"""
        combined_feedback = ""
        if feedback and feedback not in ("", "OK"):
            combined_feedback += f"FEEDBACK DE ESTILO:\n{feedback}\n\n"
        if fact_errors and fact_errors not in ("", "{}"):
            combined_feedback += f"ERROS FACTUAIS A CORRIGIR:\n{fact_errors}\n\n"

        user_content = f"""
TÓPICO: {topic}

ESBOÇO:
{outline}

PESQUISA:
{research}

ENTIDADES VERIFICADAS (use APENAS estas):
{entities}

{combined_feedback}
Escreva o artigo completo, corrigido e melhorado.
"""
    else:
        print("--- 3a. REDAÇÃO INICIAL ---")
        system_content = f"""
Você é um jornalista especialista em futebol com 20 anos de experiência.

{ANTI_HALLUCINATION}
{WRITING_STYLE}

Escreva um artigo COMPLETO, DETALHADO e ENVOLVENTE.
- Mínimo de 1200 palavras.
- Contextualize cada facto com análise e narrativa jornalística.
- Use os dados das entidades verificadas para enriquecer o texto com nomes e números reais.
- Explore cada secção do esboço com profundidade e detalhe.
"""
        user_content = f"""
TÓPICO: {topic}

ESBOÇO (siga rigorosamente):
{outline}

PESQUISA (base factual):
{research}

ENTIDADES VERIFICADAS (use APENAS estas para nomes e dados):
{entities}

Escreva o artigo completo agora.
"""

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_content),
        HumanMessage(content=user_content),
    ])

    content = (prompt | llm_writer).invoke({}).content
    time.sleep(0.5)

    return {
        "content":         content,
        "revision_needed": False,
        "review_feedback": "",
        "fact_errors":     "",
        "revision_count":  revision_count,
    }

# ====================================================================
# 4. REVIEW NODE
# ====================================================================
def review_content(state: ContentAgentState) -> dict:
    revision_count = state.get("revision_count", 0)
    print(f"--- 4. REVISÃO DE ESTILO (Tentativa {revision_count}) ---")
    content    = state["content"]
    word_count = len(content.split())
    print(f"   Palavras estimadas: {word_count}")

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Você é um revisor editorial rigoroso. Responda APENAS em JSON válido.

Critérios:
- word_count: ≥ 800 palavras?
- prose_quality: Prosa corrida (sem bullet lists como corpo)?
- grammar_spelling: Sem erros gramaticais/ortográficos?
- tone: Tom jornalístico e profissional?
- structure: Secções bem organizadas e com títulos?
- no_meta_text: Sem excesso de notas tipo "informação não disponível"?

Status OK apenas se TODOS os critérios passarem.
"""),
        HumanMessage(content=f"""
Avalie o artigo:

{content}

Retorne JSON:
{{
  "status": "OK" ou "REVISAR",
  "word_count_estimate": 0,
  "scores": {{
    "word_count": "OK|REVISAR",
    "prose_quality": "OK|REVISAR",
    "grammar_spelling": "OK|REVISAR",
    "tone": "OK|REVISAR",
    "structure": "OK|REVISAR",
    "no_meta_text": "OK|REVISAR"
  }},
  "issues": [
    {{
      "type": "tipo",
      "description": "descrição objectiva e específica",
      "suggestion": "como corrigir"
    }}
  ]
}}
"""),
    ])

    result  = (prompt | llm_fast).invoke({}).content.strip()
    time.sleep(0.5)

    MAX_REVISIONS = 3

    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    parsed = None
    if json_match:
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    if parsed is None:
        print("   ⚠️ JSON inválido no review.")
        if revision_count >= MAX_REVISIONS:
            return {"revision_needed": False, "review_feedback": "OK"}
        return {
            "revision_needed": True,
            "review_feedback": '{"issues":[{"type":"system","description":"Formato inválido","suggestion":"Rever estrutura"}]}',
        }

    if revision_count >= MAX_REVISIONS:
        print("   Limite de revisões atingido.")
        return {"revision_needed": False, "review_feedback": "OK"}

    if parsed.get("status") == "OK":
        print("   ✅ Aprovado (estilo).")
        return {"revision_needed": False, "review_feedback": "OK"}

    issues = parsed.get("issues", [])
    print(f"   🔁 {len(issues)} problema(s) de estilo.")
    for iss in issues:
        print(f"      [{iss.get('type')}] {iss.get('description')}")

    return {
        "revision_needed": True,
        "review_feedback": json.dumps(parsed, ensure_ascii=False),
    }

# ====================================================================
# 5. FACT-CHECK NODE
# ====================================================================
def fact_check_content(state: ContentAgentState) -> dict:
    print("--- 5. FACT-CHECK: Verificando factos contra fontes reais ---")
    content           = state["content"]
    raw_sources       = state.get("raw_sources", "")
    verified_entities = state.get("verified_entities", "{}")
    revision_count    = state.get("revision_count", 0)

    grounding = raw_sources[:10000] if raw_sources else verified_entities

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
Você é um verificador de factos jornalístico com acesso ao texto ORIGINAL das fontes.

Para cada afirmação ESPECÍFICA do artigo (nomes, cargos, historial, valores, datas, transferências):

1. CONFIRMADO — aparece explicitamente nas fontes. NÃO reportar.
2. INVENTADO   — nome, facto ou valor não aparece em NENHUMA fonte.
3. ERRADO      — contradiz directamente o que está nas fontes.

Regras:
- Só reporte erros tipo INVENTADO ou ERRADO.
- Afirmações genéricas sem verificação possível → ignore.
- Nome de pessoa/jogador sem estar nas entidades verificadas → INVENTADO.
- Valores financeiros sem fonte → INVENTADO.

Responda APENAS em JSON válido.
"""),
        HumanMessage(content=f"""
ENTIDADES VERIFICADAS (verdade confirmada):
{verified_entities}

FONTES ORIGINAIS:
{grounding}

ARTIGO:
{content}

Retorne JSON:
{{
  "factual_errors": [
    {{
      "claim": "afirmação exacta do artigo",
      "type": "INVENTADO|ERRADO",
      "correction": "o que as fontes dizem",
      "source": "Fonte N ou 'nenhuma'"
    }}
  ],
  "confirmed_facts_count": 0,
  "error_count": 0,
  "score": 1.0,
  "status": "OK|REVISAR"
}}

Se não houver erros INVENTADO/ERRADO:
{{"factual_errors": [], "confirmed_facts_count": 10, "error_count": 0, "score": 1.0, "status": "OK"}}
"""),
    ])

    result  = (prompt | llm_fast).invoke({}).content.strip()
    time.sleep(0.5)

    json_match = re.search(r'\{.*\}', result, re.DOTALL)
    parsed = None
    if json_match:
        try:
            parsed = json.loads(json_match.group())
        except json.JSONDecodeError:
            pass

    if parsed is None:
        print("   ⚠️ JSON inválido no fact-check. Aprovando por defeito.")
        return {"revision_needed": False, "review_feedback": "OK", "fact_errors": ""}

    real_errors = [
        e for e in parsed.get("factual_errors", [])
        if e.get("type") in ("INVENTADO", "ERRADO")
    ]
    score = parsed.get("score", 1.0)

    if real_errors:
        print(f"   ❌ {len(real_errors)} erro(s) factual(is) real(is):")
        for err in real_errors:
            print(f"      [{err['type']}] \"{err['claim'][:80]}\" → {err['correction'][:60]}")
    else:
        print(f"   ✅ Fact-check aprovado (score: {score:.2f}).")

    if real_errors and revision_count < 3:
        return {
            "revision_needed": True,
            "review_feedback": "",
            "fact_errors":     json.dumps({"errors": real_errors}, ensure_ascii=False),
        }

    if real_errors:
        print("   Limite de revisões atingido após fact-check.")

    return {"revision_needed": False, "review_feedback": "OK", "fact_errors": ""}

# ====================================================================
# 6. ROUTING
# ====================================================================
def after_review(state: ContentAgentState) -> str:
    return "draft" if state.get("revision_needed", False) else "fact_check"

def after_fact_check(state: ContentAgentState) -> str:
    if state.get("revision_needed", False) and state.get("revision_count", 0) < 3:
        return "draft"
    return "publish"

# ====================================================================
# 7. GRAPH
# ====================================================================
def create_agent_graph():
    workflow = StateGraph(ContentAgentState)

    workflow.add_node("research",         research_topic)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("plan",             plan_content)
    workflow.add_node("draft",            draft_content)
    workflow.add_node("review",           review_content)
    workflow.add_node("fact_check",       fact_check_content)

    workflow.set_entry_point("research")
    workflow.add_edge("research",         "extract_entities")
    workflow.add_edge("extract_entities", "plan")
    workflow.add_edge("plan",             "draft")
    workflow.add_edge("draft",            "review")

    workflow.add_conditional_edges(
        "review",
        after_review,
        {"draft": "draft", "fact_check": "fact_check"},
    )
    workflow.add_conditional_edges(
        "fact_check",
        after_fact_check,
        {"draft": "draft", "publish": END},
    )

    return workflow.compile()

# ====================================================================
# 8. EXECUÇÃO
# ====================================================================
if __name__ == "__main__":
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_PROJECT",    "Content-Generator-Agent")

    missing = [
        k for k in ("OPENAI_API_KEY", "LANGCHAIN_API_KEY", "SERPER_API_KEY")
        if not os.getenv(k)
    ]
    if missing:
        print(f"ERRO: Variáveis em falta: {', '.join(missing)}")
        exit(1)

    app   = create_agent_graph()
    topic = input("Insira o tópico: ").strip()
    if not topic:
        print("Nenhum tópico fornecido.")
        exit(1)

    print(f"\nINICIANDO: {topic}\n{'=' * 60}\n")

    final_state = app.invoke({
        "topic":             topic,
        "outline":           "",
        "content":           "",
        "revision_needed":   False,
        "review_feedback":   "",
        "revision_count":    0,
        "research_data":     "",
        "raw_sources":       "",
        "verified_entities": "{}",
        "fact_errors":       "",
    })

    content = final_state["content"]

    print("\n" + "#" * 60)
    print("CONTEÚDO FINAL")
    print("#" * 60)
    print(content)
    print(f"\n📊 Palavras aproximadas: {len(content.split())}")

    safe_name   = re.sub(r'[^\w\s-]', '', topic)[:50].strip().replace(" ", "_")
    output_dir  = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{safe_name}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"📄 Artigo guardado em: {output_file}")