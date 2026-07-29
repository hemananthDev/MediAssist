# 🏥 MediAssist – Healthcare AI Chatbot

An AI-powered healthcare information chatbot built with **Streamlit**, **Groq**, **ChromaDB**, **nomic-embed-text** (via Ollama), and **LangChain**. Uses Retrieval-Augmented Generation (RAG) to answer general healthcare questions grounded in a curated knowledge base.

> ⚠️ **Disclaimer:** MediAssist is for educational purposes only. It does not provide medical diagnoses and does not replace professional medical advice. Always consult a qualified healthcare provider for personal health concerns.

---

## Features

- 💬 **Conversational chat** with context-aware memory (last 6 exchanges)
- 🔍 **RAG pipeline** — answers grounded in a healthcare knowledge base via ChromaDB
- 🧠 **Groq LLM** (Llama 3 70B) for fast, high-quality responses
- 📎 **Source citations** — shows which knowledge base document was used
- 🚨 **Emergency guardrails** — detects life-threatening queries and routes to emergency services
- 🚫 **Off-topic filter** — politely redirects non-healthcare questions
- 🔄 **New conversation** button to reset chat history
- 📚 **Suggested questions** in the sidebar for easy exploration

---

## Tech Stack

| Component        | Technology                          |
|-----------------|-------------------------------------|
| Frontend        | Streamlit                           |
| LLM             | Groq API (Llama 3 70B)              |
| Embeddings      | nomic-embed-text via Ollama (local) |
| Vector Store    | ChromaDB (persisted locally)        |
| Orchestration   | LangChain                           |
| Memory          | ConversationBufferWindowMemory      |

---

## Project Structure

```
Healthcare ChatBot/
├── app.py                  # Streamlit UI
├── chatbot.py              # Core RAG chatbot logic
├── ingest.py               # Knowledge base ingestion script
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── .env                    # Your secrets (create from .env.example)
├── data/                   # Healthcare knowledge base documents
│   ├── symptoms_diseases.txt
│   ├── nutrition_lifestyle.txt
│   └── first_aid.txt
└── vectorstore/
    └── chroma_db/          # Persisted ChromaDB (auto-created by ingest.py)
```

---

## Prerequisites

- Python 3.10 or higher
- [Ollama](https://ollama.com/download) installed and running
- A free [Groq API key](https://console.groq.com)

---

## Setup Instructions

### 1. Clone or download the project

```bash
cd "Healthcare ChatBot"
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
pip install -r requirements.txt
```

### 4. Pull the embedding model via Ollama

Make sure Ollama is running, then pull the embedding model:

```bash
ollama pull nomic-embed-text
```

### 5. Configure environment variables

```bash
# Windows
copy .env.example .env

# macOS / Linux
cp .env.example .env
```

Open `.env` and add your Groq API key:

```
GROQ_API_KEY=your_actual_groq_api_key_here
GROQ_MODEL=llama3-70b-8192
```

### 6. Install dependencies

```bash
python run.py --setup
```

### 7. Build the knowledge base (run once)

```bash
python run.py --ingest
```

### 8. Launch the chatbot

```bash
python run.py
```

The app opens at `http://localhost:8501` in your browser.

> You can also run each step manually:
> ```bash
> python ingest.py          # build knowledge base
> streamlit run app.py      # launch UI directly
> ```

---

## Usage

1. Type a healthcare question in the input box, or click a suggested question in the sidebar.
2. MediAssist retrieves relevant content from the knowledge base and generates a response.
3. Sources used are shown below each response.
4. Click **New Conversation** in the sidebar to reset the chat history.

### Example questions

- *"What are the early signs of diabetes?"*
- *"How can I manage high blood pressure without medication?"*
- *"What should I do if someone is choking?"*
- *"What foods are good for heart health?"*
- *"How do I treat a minor burn at home?"*

---

## Adding to the Knowledge Base

To add more healthcare content:

1. Create a new `.txt` file in the `data/` directory.
2. Re-run the ingestion script: `python ingest.py`
3. Restart the Streamlit app.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `FileNotFoundError: Vectorstore not found` | Run `python ingest.py` first |
| `GROQ_API_KEY is not set` | Check your `.env` file has a valid key |
| `Connection refused` (Ollama) | Start Ollama: run `ollama serve` in a terminal |
| Slow first load | ChromaDB is initialising; subsequent loads are faster |

---

## Architecture Overview

```
User Query
    │
    ▼
[Streamlit UI] ──► [Guardrails: Emergency / Off-topic check]
    │                       │
    │                       ▼ (if safe)
    │              [LangChain ConversationalRetrievalChain]
    │                       │
    │              ┌─────────────────────┐
    │              │  ChromaDB Retriever │◄── nomic-embed-text (Ollama)
    │              │  (Top-4 chunks)     │
    │              └─────────────────────┘
    │                       │
    │              [Groq LLM: Llama 3 70B]
    │              [+ Conversation Memory]
    │                       │
    └──────────────── Response + Sources
```
