# 🏥 MediAssist – Healthcare AI Chatbot

An AI-powered healthcare information chatbot built with **Streamlit**, **Groq**, **ChromaDB**, **nomic-embed-text** (via Ollama), and **LangChain**. Uses Retrieval-Augmented Generation (RAG) to answer general healthcare questions grounded in a curated knowledge base.

> ⚠️ **Disclaimer:** MediAssist is for educational purposes only. It does not provide medical diagnoses and does not replace professional medical advice. Always consult a qualified healthcare provider for personal health concerns.

---

## Features

- 💬 **Conversational chat** with context-aware memory (last 6 exchanges)
- 🔍 **RAG pipeline** — answers grounded in a healthcare knowledge base via ChromaDB
- 🧠 **Groq LLM** (llama-3.3-70b-versatile) for fast, high-quality responses
- 📎 **Source citations** — shows which knowledge base document was used
- 🟢 **Confidence indicator** — High / Medium / Low based on retrieval cosine distance
- 🚨 **Emergency guardrails** — detects life-threatening queries and routes to emergency services
- 💊 **Medication guardrail** — blocks advice on changing/stopping prescriptions
- 🔬 **Diagnosis guardrail** — refuses personal disease diagnosis requests
- 🚫 **Off-topic filter** — politely redirects non-healthcare questions
- 🔄 **New conversation** button to reset chat history
- 📚 **Suggested questions** in the sidebar
- 🌙 **Dark theme** UI

---

## Tech Stack

| Component        | Technology                              |
|-----------------|-----------------------------------------|
| Frontend        | Streamlit (dark theme)                  |
| LLM             | Groq API (llama-3.3-70b-versatile)      |
| Embeddings      | nomic-embed-text via Ollama (local)     |
| Vector Store    | ChromaDB (persisted locally)            |
| Orchestration   | LangChain LCEL (`prompt | llm | parser`)|
| Conversation Memory | Python list (manual window of 6 exchanges) |
| Guardrails      | Regex + keyword matching (`src/utils.py`) |

---

## Project Structure

```
MediAssist/
├── app.py                      # Streamlit UI
├── chatbot.py                  # Orchestrator: guardrails → retrieval → LLM
├── ingest.py                   # Knowledge base ingestion script
├── run.py                      # Single-command launcher with pre-flight checks
├── requirements.txt
├── .env.example
├── .streamlit/
│   └── config.toml             # Dark theme config
├── src/
│   ├── prompts.py              # System prompt + canned guardrail responses
│   ├── rag.py                  # ChromaRetriever with confidence scoring
│   └── utils.py                # All guardrail check functions
├── data/                       # TXT knowledge base documents
│   ├── symptoms_diseases.txt
│   ├── nutrition_lifestyle.txt
│   └── first_aid.txt
├── pdfs/                       # PDF knowledge base (WHO, CDC documents)
├── assets/
└── vectorstore/
    └── chroma_db/              # Persisted ChromaDB (built by ingest.py)
```

---

## Architecture

```
User Query
    │
    ▼
[Streamlit UI]
    │
    ▼
[Guardrail Layer — src/utils.py]
    ├── Emergency?      → Immediate emergency response
    ├── Safety violation? → Rejection response
    ├── Greeting?       → Friendly intro response
    ├── Diagnosis req?  → Refer to professional
    ├── Medication chg? → Refer to physician
    └── Off-topic?      → Polite redirect
    │
    ▼ (passes guardrails)
[ChromaRetriever — src/rag.py]
    │  ollama.embed(nomic-embed-text) → cosine similarity search
    │  Returns top-4 chunks + cosine distance → confidence label
    ▼
[LangChain LCEL Chain — chatbot.py]
    │  ChatPromptTemplate (system + chat_history + question)
    │  ChatGroq (llama-3.3-70b-versatile)
    │  StrOutputParser
    ▼
[Manual conversation memory]
    │  Python list of HumanMessage / AIMessage (last 6 exchanges)
    ▼
[Response + Sources + Confidence]
    │
    ▼
[Streamlit UI — renders chat bubble, badge, source tags]
```

---

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed and running locally
- A free [Groq API key](https://console.groq.com)

> **Note on Ollama:** MediAssist uses `nomic-embed-text` locally via Ollama for embeddings — this avoids any embedding API cost and keeps document data on-device. You must have Ollama running before ingestion or launching the app.

---

## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/hemananthDev/MediAssist.git
cd MediAssist
```

### 2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
python run.py --setup
```

### 4. Start Ollama and pull the embedding model

```bash
ollama serve                      # start Ollama (keep this terminal open)
ollama pull nomic-embed-text      # pull embedding model (one-time)
```

### 5. Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
```

### 6. Build the knowledge base

```bash
python run.py --ingest
```

This embeds all documents in `data/` and `pdfs/` into ChromaDB (~10-15 min for full PDF set).

### 7. Launch the chatbot

```bash
python run.py
```

App opens at `http://localhost:8501`.

> **Manual alternative:**
> ```bash
> python ingest.py        # build knowledge base
> streamlit run app.py    # launch UI
> ```

---

## Usage

1. Type a question or click a suggested question in the sidebar.
2. MediAssist retrieves relevant content from the knowledge base and generates a response.
3. Each response shows a **confidence badge** (🟢 High / 🟡 Medium / 🔴 Low) and **source citations**.
4. Click **New Conversation** to reset chat history.

### Example questions

- *"What are the early signs of diabetes?"*
- *"How can I manage high blood pressure without medication?"*
- *"What should I do if someone is choking?"*
- *"What foods are good for heart health?"*
- *"How do I treat a minor burn at home?"*
- *"What vitamins should I take for immunity?"*

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Vectorstore not found` | Run `python run.py --ingest` first |
| `GROQ_API_KEY is not set` | Check your `.env` file has a valid key |
| `ConnectionError: Failed to connect to Ollama` | Run `ollama serve` in a separate terminal |
| `model_decommissioned` error | Set `GROQ_MODEL=llama-3.3-70b-versatile` in `.env` |
| Slow first load | ChromaDB initialising; subsequent loads are faster |
