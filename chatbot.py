"""
chatbot.py
----------
Core chatbot interface.
Orchestrates retrieval (src/rag.py), guardrails (src/utils.py),
prompt engineering (src/prompts.py), and the Groq LLM chain.
"""

import os
import logging
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.rag import ChromaRetriever
from src.prompts import (
    build_prompt,
    EMERGENCY_RESPONSE,
    DIAGNOSIS_RESPONSE,
    MEDICATION_RESPONSE,
    OFF_TOPIC_RESPONSE,
    SAFETY_RESPONSE,
    FALLBACK_RESPONSE,
)
from src.utils import (
    is_emergency,
    is_diagnosis_request,
    is_medication_change_request,
    is_off_topic,
    is_safety_violation,
    is_low_confidence_answer,
    is_greeting,
)

# ── Environment ────────────────────────────────────────────────────────────────
load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent
VECTORSTORE_DIR = BASE_DIR / "vectorstore" / "chroma_db"
GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
MAX_HISTORY     = 6    # max past exchange pairs kept in memory


def build_chain(temperature: float = 0.3):
    """
    Build the RAG chain using Groq LLM.
    Temperature: 0.0 (deterministic) to 1.0 (creative).
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY is not set. Add it to your .env file.")

    llm = ChatGroq(
        api_key=groq_api_key,
        model_name=GROQ_MODEL,
        temperature=temperature,
        max_tokens=1024,
    )

    prompt = build_prompt()

    chain = (
        RunnablePassthrough()
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


class HealthChatbot:
    """
    Main chatbot interface.
    Manages retrieval, guardrails, memory, and LLM invocation.
    """

    def __init__(self, temperature: float = 0.3):
        self._retriever    = ChromaRetriever(VECTORSTORE_DIR)
        self._chain        = build_chain(temperature)
        self._temperature  = temperature
        self._chat_history: list = []

    def set_temperature(self, temperature: float):
        """Update LLM temperature (rebuilds the chain)."""
        self._temperature = temperature
        self._chain = build_chain(temperature)

    def chat(self, user_message: str) -> dict:
        """
        Process a user message and return:
            answer     : str   – assistant response
            sources    : list  – unique source filenames
            confidence : str   – "High" | "Medium" | "Low"
            category   : str   – "emergency" | "diagnosis" | "medication" | 
                                 "off_topic" | "safety" | "healthcare"
        """
        # ── Guardrails (ordered by priority) ──────────────────────────────
        if is_emergency(user_message):
            return {
                "answer":     EMERGENCY_RESPONSE,
                "sources":    [],
                "confidence": "N/A",
                "category":   "emergency",
            }

        if is_safety_violation(user_message):
            return {
                "answer":     SAFETY_RESPONSE,
                "sources":    [],
                "confidence": "N/A",
                "category":   "safety",
            }

        if is_greeting(user_message):
            return {
                "answer":     (
                    "Hello! I'm MediAssist, your AI healthcare information assistant. "
                    "I can help you with questions about symptoms, diseases, nutrition, "
                    "lifestyle, preventive care, and first aid.\n\n"
                    "What healthcare topic can I help you with today?"
                ),
                "sources":    [],
                "confidence": "N/A",
                "category":   "greeting",
            }

        if is_diagnosis_request(user_message):
            return {
                "answer":     DIAGNOSIS_RESPONSE,
                "sources":    [],
                "confidence": "N/A",
                "category":   "diagnosis",
            }

        if is_medication_change_request(user_message):
            return {
                "answer":     MEDICATION_RESPONSE,
                "sources":    [],
                "confidence": "N/A",
                "category":   "medication",
            }

        if is_off_topic(user_message):
            return {
                "answer":     OFF_TOPIC_RESPONSE,
                "sources":    [],
                "confidence": "N/A",
                "category":   "off_topic",
            }

        # ── Retrieve relevant chunks ──────────────────────────────────────
        chunks  = self._retriever.retrieve(user_message)
        context = self._retriever.format_context(chunks)
        sources = self._retriever.unique_sources(chunks)

        # Top chunk confidence (based on cosine distance)
        retrieval_confidence = self._retriever.top_confidence(chunks)

        # ── Trim chat history to last MAX_HISTORY exchanges ───────────────
        trimmed_history = self._chat_history[-(MAX_HISTORY * 2):]

        # ── Invoke LLM ─────────────────────────────────────────────────────
        answer = self._chain.invoke({
            "context":      context,
            "chat_history": trimmed_history,
            "question":     user_message,
        })

        # ── Detect if LLM couldn't answer (fallback confidence) ────────────
        if is_low_confidence_answer(answer):
            answer = FALLBACK_RESPONSE
            final_confidence = "Low"
        else:
            final_confidence = retrieval_confidence

        # ── Update memory ──────────────────────────────────────────────────
        self._chat_history.append(HumanMessage(content=user_message))
        self._chat_history.append(AIMessage(content=answer))

        return {
            "answer":     answer,
            "sources":    sources,
            "confidence": final_confidence,
            "category":   "healthcare",
        }

    def clear_memory(self):
        """Reset conversation memory."""
        self._chat_history = []
