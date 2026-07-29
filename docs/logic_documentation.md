# MediAssist – Logic Documentation

**Project:** AI-Powered Healthcare Chatbot  
**Position:** AI Engineer (Contractual)  
**Author:** Hemanth  
**Stack:** Streamlit · Groq (llama-3.3-70b-versatile) · LangChain LCEL · ChromaDB · nomic-embed-text (Ollama)

---

## 1. How the Chatbot Processes User Queries

Every user message passes through a strict sequential pipeline before a response is returned. The pipeline has two major stages: **Guardrail Evaluation** and **RAG Generation**.

### 1.1 Full Query Processing Flow

```
User Message
      │
      ▼
┌─────────────────────────────────────────────────────┐
│                  GUARDRAIL LAYER                    │
│  Evaluated in priority order (chatbot.py):          │
│                                                     │
│  1. is_emergency()     → 911 / emergency response   │
│  2. is_safety_violation() → rejection               │
│  3. is_greeting()      → friendly intro, skip RAG   │
│  4. is_diagnosis_request() → refer to professional  │
│  5. is_medication_change_request() → refer to doctor│
│  6. is_off_topic()     → polite redirect            │
└─────────────────────────────────────────────────────┘
      │ (passes all guardrails)
      ▼
┌─────────────────────────────────────────────────────┐
│               RAG RETRIEVAL (src/rag.py)            │
│                                                     │
│  query → ollama.embed(nomic-embed-text)             │
│        → ChromaDB cosine similarity search          │
│        → Top-4 chunks returned with distances       │
│        → Confidence label assigned per chunk        │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│            LLM CHAIN (chatbot.py + src/prompts.py)  │
│                                                     │
│  ChatPromptTemplate:                                │
│    system prompt + chat_history + question          │
│        → ChatGroq (llama-3.3-70b-versatile)         │
│        → StrOutputParser                            │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│            POST-PROCESSING                          │
│                                                     │
│  is_low_confidence_answer() checks first 15 words   │
│  → if LLM admits it doesn't know → FALLBACK_RESPONSE│
│  → else → return answer + sources + confidence      │
└─────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────┐
│            MEMORY UPDATE                            │
│                                                     │
│  HumanMessage + AIMessage appended to list          │
│  Trimmed to last 6 exchanges (12 messages)          │
└─────────────────────────────────────────────────────┘
      │
      ▼
  Response dict: { answer, sources, confidence, category }
```

### 1.2 Conversation Memory

Memory is maintained as a plain Python list of `HumanMessage` and `AIMessage` objects (LangChain Core types). The last 6 exchange pairs (12 messages) are injected into the prompt via `MessagesPlaceholder`. This gives the LLM enough context for follow-up questions without inflating the token count.

Memory is reset when the user clicks **New Conversation** in the sidebar, which calls `HealthChatbot.clear_memory()`.

---

## 2. Knowledge Base and Retrieval

### 2.1 Knowledge Base

The knowledge base consists of two source types:

| Source | Files | Content |
|--------|-------|---------|
| TXT (curated) | `symptoms_diseases.txt`, `nutrition_lifestyle.txt`, `first_aid.txt` | Structured healthcare reference content written specifically for this chatbot |
| PDF (authoritative) | 11 documents including WHO manuals, CDC fact sheets, hypertension guide, type-2 diabetes guide, mental health fact sheet, first aid manual | Published healthcare authority documents |

Total: **615 pages / documents → 2,919 chunks** after splitting.

### 2.2 Ingestion Pipeline (ingest.py)

```
TXT files (data/)  +  PDF files (pdfs/)
           │
    Document loading
    (DirectoryLoader / PyPDFLoader)
           │
    RecursiveCharacterTextSplitter
    chunk_size=800, overlap=100
    separators: [\n\n, \n, ., space]
           │
    Filter: discard chunks < 50 chars
           │
    ollama.embed(nomic-embed-text)
    Batches of 32 — one HTTP call per batch
    (avoids Windows socket exhaustion)
           │
    chromadb.PersistentClient
    Collection: healthcare_kb
    Distance metric: cosine
```

The ingestion is designed to be idempotent — re-running `python ingest.py` wipes and rebuilds the vectorstore cleanly using `shutil.rmtree` (cross-platform).

### 2.3 Retrieval (src/rag.py)

At query time:
1. The user query is embedded using `ollama.embed(nomic-embed-text)` — same model as ingestion, ensuring vector space consistency.
2. ChromaDB performs cosine similarity search, returning the top 4 chunks with their distance scores.
3. Each chunk is assigned a confidence label based on cosine distance thresholds:
   - **High**: distance ≤ 0.25
   - **Medium**: distance ≤ 0.45
   - **Low**: distance > 0.45
4. The top-1 chunk's confidence label is used as the overall response confidence shown in the UI.

---

## 3. How Prompts Are Customized

### 3.1 System Prompt Design

The system prompt (`src/prompts.py`) is injected as the first message in every LLM call. It contains 9 explicit behavioral rules:

| Rule | Purpose |
|------|---------|
| Answer only healthcare questions | Domain restriction |
| Use retrieved context as primary source | Grounds answers in the knowledge base |
| Fallback phrase if context is missing | Prevents hallucination |
| Never diagnose diseases | Core safety constraint |
| Never prescribe medications | Core safety constraint |
| Diagnosis-request response | Explicit wording for refusal |
| Medication-change response | Explicit wording for refusal |
| Keep answers concise and jargon-free | Readability |
| End every response with disclaimer | Compliance / assignment requirement |

The retrieved context is injected directly into the system message as `{context}`, placing it at the highest-priority position in the prompt rather than as a separate user message.

### 3.2 Prompt Template Structure

```
SystemMessage:
    [9-rule system prompt]
    [retrieved context — top 4 chunks]

MessagesPlaceholder: chat_history
    [last 6 HumanMessage / AIMessage pairs]

HumanMessage:
    [current user question]
```

### 3.3 Temperature Control

The LLM temperature is configurable from the Streamlit sidebar (0.0–1.0, default 0.3). At 0.3, responses are factual and consistent while allowing natural language variation. This maps directly to `ChatGroq(temperature=temperature)` and rebuilds the chain via `HealthChatbot.set_temperature()`.

---

## 4. Safety Measures and Validation

### 4.1 Guardrail System (src/utils.py)

Six guardrail categories are evaluated before any LLM call:

| Category | Detection Method | Response |
|----------|-----------------|----------|
| **Emergency** | Keyword substring match (broad — life safety priority) | Direct emergency services instruction |
| **Safety violation** | Keyword substring match (explicit harmful intent) | Rejection |
| **Greeting** | Regex anchored patterns (`^hi$`, `^hello$`, etc.) | Warm intro, no RAG |
| **Diagnosis request** | Regex with named condition list | Refer to professional |
| **Medication change** | Regex with filler-word tolerance (`(\w+\s+){0,3}`) | Refer to physician |
| **Off-topic** | Keyword substring (unambiguous non-health domains only) | Polite redirect |

**Design principle:** The guardrails prefer false negatives over false positives. A borderline question that slips through to the LLM is handled by the system prompt rules. A wrongly blocked legitimate healthcare question is a worse outcome.

The medication and diagnosis guardrails use regex patterns rather than simple substring matching to avoid false positives on common healthcare phrases such as:
- *"What vitamins should I take?"* — allowed (not a medication change)
- *"Do I have to avoid dairy?"* — allowed (not a diagnosis request)
- *"I hurt my ankle playing football"* — allowed (not off-topic)

### 4.2 Response Validation

After the LLM responds, `is_low_confidence_answer()` inspects the **first 15 words** of the response for phrases indicating the model admitted it couldn't find an answer (e.g., *"I couldn't find reliable information..."*). If detected, the response is replaced with the standardized `FALLBACK_RESPONSE`. Checking only the first 15 words prevents discarding valid responses that contain hedging language mid-sentence.

### 4.3 Error Handling

All exceptions in the Streamlit UI are caught and replaced with a generic user-friendly message. Raw Python exception text is never shown to the user. Errors are logged internally via Python's `logging` module.

### 4.4 Mandatory Disclaimer

The system prompt rule #9 requires the LLM to append the following disclaimer to every medical response:

> *"⚠️ This information is for educational purposes only and is not a substitute for professional medical advice."*

A static disclaimer banner is also permanently displayed at the top of the UI, independent of LLM behavior.

---

## 5. Assumptions Made During Development

1. **Ollama is a local dependency.** The system requires Ollama to be running locally to generate embeddings. This was a deliberate choice to eliminate embedding API costs and keep document data on-device. The `run.py` pre-flight check verifies Ollama is running before launching.

2. **Confidence score is retrieval-based, not answer-grounded.** The High/Medium/Low confidence indicator reflects how closely the top retrieved chunk matched the query by cosine distance — it does not measure whether the LLM's answer is factually correct. A chunk can be semantically close while tangentially relevant.

3. **Memory is session-scoped and in-memory only.** Conversation history is not persisted to a database. Restarting the app clears all history. This is appropriate for a demo/assessment context.

4. **The knowledge base is static.** No real-time medical data sources are queried. All answers are grounded in the documents ingested at setup time. Adding new documents requires re-running `python ingest.py`.

5. **The chatbot targets general health education, not clinical decision support.** It is designed for users seeking general health information, not for clinical use. All guardrails and disclaimers reflect this scope.

6. **PDF content quality varies.** Some PDFs (particularly older WHO documents) use non-standard encodings that pypdf cannot fully parse. These generate harmless `Advanced encoding` warnings during ingestion. The readable portions of those documents are still indexed.

---

## 6. Tech Stack Rationale

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **LLM** | Groq API — llama-3.3-70b-versatile | Free tier, extremely fast inference (<1s), Meta's latest open-weight model, strong instruction following |
| **Embeddings** | nomic-embed-text via Ollama | Local execution, no API cost, 768-dimensional embeddings competitive with paid alternatives, same model at ingest and query time guarantees vector consistency |
| **Vector DB** | ChromaDB | Lightweight, file-persisted, no server process required, cosine similarity built-in, sufficient for 3,000-chunk knowledge base |
| **Framework** | LangChain LCEL | `prompt \| llm \| parser` chain is readable and maintainable; native integration with ChatGroq and prompt templates |
| **UI** | Streamlit | Preferred by the assignment; rapid prototyping; session state for memory; no separate backend needed |
| **Launcher** | Custom `run.py` | Pre-flight checks (Python version, Ollama, API key, vectorstore) catch setup errors before they reach the user |
