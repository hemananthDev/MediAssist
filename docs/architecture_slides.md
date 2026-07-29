# MediAssist – Architecture Presentation
## AI-Powered Healthcare Chatbot
### AI Engineer Technical Assignment

> **Instructions for use:**
> Copy each slide's content into Google Slides or PowerPoint.
> Suggested theme: dark background (#0b0e14), accent color (#4dabf7), white text.
> 5 slides + title = 6 total (title counts as slide 1).

---

---

# SLIDE 1 — Title

## 🏥 MediAssist
### AI-Powered Healthcare Chatbot

**Built with:**
Streamlit · Groq · LangChain · ChromaDB · Ollama

---
*Candidate: Hemanth*
*Position: AI Engineer (Contractual)*

---

---

# SLIDE 2 — System Architecture

## Overall System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER                                 │
└───────────────────────┬─────────────────────────────────────┘
                        │ query
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                  STREAMLIT UI  (app.py)                     │
│   Dark theme · Chat bubbles · Confidence badge · Sources    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              GUARDRAIL LAYER  (src/utils.py)                │
│                                                             │
│  Emergency → Safety → Greeting → Diagnosis →               │
│  Medication Change → Off-topic                              │
│                                                             │
│  Canned responses returned immediately if triggered        │
└───────────────────────┬─────────────────────────────────────┘
                        │ (safe query)
          ┌─────────────┴──────────────┐
          ▼                            ▼
┌──────────────────┐       ┌───────────────────────┐
│   CHROMADB       │       │  CONVERSATION MEMORY  │
│   RETRIEVER      │       │                       │
│  (src/rag.py)    │       │  Python list          │
│                  │       │  Last 6 exchanges     │
│  nomic-embed-text│       │  HumanMessage +       │
│  Top-4 chunks    │       │  AIMessage            │
│  + confidence    │       └───────────────────────┘
└────────┬─────────┘                   │
         └─────────────┬───────────────┘
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LLM CHAIN  (chatbot.py)                        │
│                                                             │
│  ChatPromptTemplate  →  ChatGroq  →  StrOutputParser        │
│  (src/prompts.py)       Groq API      LangChain LCEL        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
              Response + Sources + Confidence
```

---

---

# SLIDE 3 — Tech Stack & LLM Rationale

## Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **UI** | Streamlit | Preferred by assignment; session state; rapid prototyping |
| **LLM** | Groq — llama-3.3-70b-versatile | Sub-second inference, free tier, Meta's best open-weight model, strong instruction following |
| **Embeddings** | nomic-embed-text (Ollama) | Local — no API cost, no data leaving device, 768-dim vectors competitive with paid alternatives |
| **Vector DB** | ChromaDB | File-persisted, no server needed, cosine similarity built-in, handles 3K chunks easily |
| **Orchestration** | LangChain LCEL | Clean `prompt \| llm \| parser` pipeline, native Groq integration |
| **Memory** | Python list | Simple, sufficient — session-scoped, no persistence overhead |
| **Launcher** | Custom run.py | Pre-flight checks: Python version, Ollama, API key, vectorstore |

### LLM Selection Rationale

**Why Groq over OpenAI/Gemini?**
- Groq's LPU (Language Processing Unit) delivers ~10x faster inference than GPU-based APIs
- Free tier supports the full demo without billing risk
- `llama-3.3-70b-versatile` scores comparably to GPT-4o on instruction-following benchmarks
- No data retention policy — appropriate for healthcare context

---

---

# SLIDE 4 — RAG Pipeline & Prompt Engineering

## RAG Pipeline

```
INGESTION (run once)                    QUERY TIME

PDF files  ──┐                          User question
TXT files  ──┤                               │
             ▼                               ▼
      PyPDFLoader /               ollama.embed(nomic-embed-text)
      TextLoader                             │
             │                               ▼
  RecursiveCharacterTextSplitter     ChromaDB cosine search
  chunk_size=800, overlap=100               │
             │                       Top-4 chunks returned
  ollama.embed(nomic-embed-text)            │
  Batches of 32 chunks               Confidence scoring:
             │                       ≤0.25 → High
  ChromaDB PersistentClient          ≤0.45 → Medium
  Collection: healthcare_kb          >0.45 → Low
  Metric: cosine                            │
                                     Injected into prompt
```

**Knowledge Base:** 615 documents → 2,919 chunks
Sources: WHO manuals, CDC fact sheets, hypertension guide, diabetes guide, mental health, first aid manual + 3 curated TXT files

## Prompt Engineering Strategy

The system prompt enforces **9 explicit behavioral rules:**

1. Answer only healthcare questions
2. Use retrieved context as primary source
3. Fallback phrase if context is missing → prevents hallucination
4. Never diagnose diseases
5. Never prescribe medications
6. Explicit wording for diagnosis refusals
7. Explicit wording for medication change refusals
8. Concise, jargon-free language
9. Mandatory disclaimer on every medical response

**Context placement:** Retrieved chunks are injected into the *system message* (highest priority position), not as a separate user message — this ensures the LLM treats knowledge base content as authoritative ground truth.

---

---

# SLIDE 5 — Guardrails, Confidence & Challenges

## Safety Guardrails

Six-tier guardrail system evaluated **before** any LLM call:

| Priority | Guardrail | Trigger | Action |
|----------|-----------|---------|--------|
| 1 | Emergency | "chest pain", "heart attack", "choking"... | 911 / emergency services instruction |
| 2 | Safety | "how to make a bomb", "poison someone"... | Hard rejection |
| 3 | Greeting | "hi", "hello", "good morning"... | Warm intro, skip RAG |
| 4 | Diagnosis | "do I have diabetes?", "diagnose me"... | Refer to professional |
| 5 | Medication | "should I stop my blood pressure medication?" | Refer to physician |
| 6 | Off-topic | "bitcoin", "weather forecast", "debug my code"... | Polite redirect |

**Design principle:** Regex-based patterns (not simple substring matching) for diagnosis and medication guardrails — prevents false positives on legitimate questions like *"What vitamins should I take?"*

## Confidence Indicator

Each response displays a confidence badge based on the cosine distance of the best-matched chunk:
- 🟢 **High** (distance ≤ 0.25) — strong match in knowledge base
- 🟡 **Medium** (distance ≤ 0.45) — partial match
- 🔴 **Low** (distance > 0.45) — weak match, answer from general LLM knowledge

## Challenges & Solutions

| Challenge | Solution |
|-----------|---------|
| Windows socket exhaustion during ingestion (2,919 chunks × per-chunk HTTP calls) | Bypassed LangChain embeddings wrapper; used native `ollama.embed()` with batch size 32 — one HTTP request per batch |
| Windows file locks on ChromaDB during re-ingestion | Replaced PowerShell subprocess with cross-platform `shutil.rmtree(ignore_errors=True)` |
| Guardrail false positives blocking legitimate healthcare questions | Replaced keyword lists with specific regex patterns; added negative test cases to verify |
| LangChain version conflicts (langchain-classic, langgraph) on existing environment | Upgraded to latest compatible versions across entire stack |

---

---

# SLIDE 6 — Application Workflow & Innovation

## Application Workflow

```
1. Setup          python run.py --setup     Install dependencies
                  ollama pull nomic-embed-text
                  Configure .env (GROQ_API_KEY)

2. Ingestion      python run.py --ingest    Load PDFs + TXT
                                            Embed → ChromaDB
                                            ~10-15 min (local CPU)

3. Launch         python run.py             Pre-flight checks:
                                            ✓ Python 3.10+
                                            ✓ Ollama running
                                            ✓ API key set
                                            ✓ Vectorstore exists
                                            → streamlit run app.py

4. Chat           http://localhost:8501      User types or clicks
                                            suggested question
                                            → Guardrail check
                                            → RAG retrieval
                                            → LLM generation
                                            → Response + badge + sources
```

## Additional Features (Innovation)

| Feature | Implementation |
|---------|---------------|
| **Confidence indicator** | Cosine distance → High/Medium/Low badge on every response |
| **Source citations** | Deduplicated filenames shown below each assistant message |
| **Temperature slider** | Sidebar slider (0.0–1.0); live chain rebuild via `set_temperature()` |
| **Automated test suite** | 106 pytest cases covering all 7 guardrail functions (true positives + false positive prevention) |
| **Pre-flight launcher** | `run.py` validates entire environment before starting |
| **Greeting handler** | Short greetings bypass RAG to avoid irrelevant document retrieval |
| **Dark theme UI** | `.streamlit/config.toml` + custom CSS — production-grade appearance |

---

*MediAssist — For educational purposes only. Not a substitute for professional medical advice.*
