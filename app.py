"""
app.py
------
Streamlit frontend for the MediAssist Healthcare Chatbot.

Run with:
    streamlit run app.py
"""

import logging
import warnings

warnings.filterwarnings("ignore")
logging.getLogger("chromadb").setLevel(logging.ERROR)

import streamlit as st
from chatbot import HealthChatbot

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MediAssist – Healthcare AI Chatbot",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Clean Modern Theme ──────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ── Reset & Base ── */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        box-sizing: border-box;
    }
    
    .stApp {
        background: #0b0e14;
    }
    
    .main .block-container {
        padding: 2rem 2rem 2rem 2rem !important;
        max-width: 820px;
        margin: 0 auto;
    }
    
    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6 {
        font-weight: 600 !important;
        letter-spacing: -0.02em !important;
        color: #f1f5f9 !important;
        margin-bottom: 0.5rem !important;
    }
    
    p, span, div, li, label {
        color: #e2e8f0 !important;
    }
    
    /* ── Hide Streamlit branding ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* ── Input field ── */
    .stTextInput > div > div > input {
        background: #141922 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2a3344 !important;
        border-radius: 12px !important;
        padding: 12px 18px !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4f8cf7 !important;
        box-shadow: 0 0 0 3px rgba(79, 140, 247, 0.15) !important;
    }
    
    .stTextInput > div > div > input::placeholder {
        color: #64748b !important;
    }
    
    /* ── Chat bubbles ── */
    .message-wrapper {
        margin-bottom: 1rem;
    }
    
    .user-bubble {
        background: linear-gradient(135deg, #1e3a5f, #1f4a7a);
        color: #f1f5f9;
        border-radius: 16px 16px 4px 16px;
        padding: 12px 18px;
        margin: 0 0 0 auto;
        max-width: 75%;
        font-size: 0.95rem;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(31, 74, 122, 0.2);
        border: 1px solid rgba(79, 140, 247, 0.08);
        animation: slideUp 0.3s ease;
    }
    
    .assistant-bubble {
        background: #141922;
        color: #e2e8f0;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 20px;
        margin: 0 0 0 0;
        max-width: 85%;
        border: 1px solid #2a3344;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        animation: slideUp 0.3s ease;
        line-height: 1.7;
    }
    
    .assistant-bubble strong {
        color: #60a5fa;
    }
    
    .emergency-bubble {
        background: #1f0f0f;
        color: #fca5a5;
        border-radius: 16px 16px 16px 4px;
        padding: 14px 20px;
        margin: 0 0 0 0;
        max-width: 85%;
        border-left: 4px solid #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.2);
        box-shadow: 0 2px 8px rgba(239, 68, 68, 0.1);
        animation: slideUp 0.3s ease;
        line-height: 1.7;
    }
    
    .emergency-bubble strong {
        color: #f87171;
    }
    
    /* ── Chat metadata ── */
    .chat-meta {
        display: flex;
        align-items: center;
        gap: 10px;
        flex-wrap: wrap;
        margin: 4px 0 4px 4px;
        padding: 2px 0;
    }
    
    .badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        padding: 3px 12px;
        border-radius: 100px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.02em;
        background: #141922;
        border: 1px solid #2a3344;
        color: #94a3b8;
    }
    
    .badge-high {
        color: #4ade80;
        border-color: rgba(74, 222, 128, 0.2);
        background: rgba(74, 222, 128, 0.05);
    }
    
    .badge-medium {
        color: #fbbf24;
        border-color: rgba(251, 191, 36, 0.2);
        background: rgba(251, 191, 36, 0.05);
    }
    
    .badge-low {
        color: #f87171;
        border-color: rgba(248, 113, 113, 0.2);
        background: rgba(248, 113, 113, 0.05);
    }
    
    .badge-na {
        color: #94a3b8;
        border-color: rgba(148, 163, 184, 0.15);
        background: rgba(148, 163, 184, 0.05);
    }
    
    .source-tag {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 100px;
        font-size: 0.7rem;
        background: rgba(79, 140, 247, 0.05);
        border: 1px solid rgba(79, 140, 247, 0.1);
        color: #60a5fa;
    }
    
    /* ── Animations ── */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(8px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* ── Disclaimer ── */
    .disclaimer-box {
        background: #141922;
        border-left: 3px solid #fbbf24;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 0.82rem;
        color: #94a3b8;
        border: 1px solid #2a3344;
        margin-bottom: 1.5rem;
        line-height: 1.6;
    }
    
    .disclaimer-box strong {
        color: #fbbf24;
    }
    
    .emergency-number {
        display: inline-block;
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 2px 12px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.9rem;
        border: 1px solid rgba(239, 68, 68, 0.2);
    }
    
    /* ── Welcome screen ── */
    .welcome-container {
        text-align: center;
        padding: 3rem 1rem;
        background: #141922;
        border-radius: 16px;
        border: 1px solid #2a3344;
        margin: 1rem 0;
    }
    
    .welcome-icon {
        font-size: 3.5rem;
        margin-bottom: 0.75rem;
    }
    
    .welcome-title {
        font-size: 2rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 0.5rem;
    }
    
    .welcome-subtitle {
        color: #94a3b8;
        font-size: 1rem;
        max-width: 500px;
        margin: 0 auto;
        line-height: 1.6;
    }
    
    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #0b0e14 !important;
        border-right: 1px solid #1a1f2e !important;
        padding: 1.5rem 1.25rem !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e2e8f0 !important;
    }
    
    section[data-testid="stSidebar"] hr {
        border-color: #1a1f2e;
        margin: 1.25rem 0;
    }
    
    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton button {
        background: #141922 !important;
        color: #e2e8f0 !important;
        border: 1px solid #2a3344 !important;
        border-radius: 8px !important;
        padding: 8px 16px !important;
        font-size: 0.85rem !important;
        transition: all 0.2s ease !important;
        width: 100% !important;
        justify-content: center !important;
    }
    
    section[data-testid="stSidebar"] .stButton button:hover {
        background: #1a2332 !important;
        border-color: #4f8cf7 !important;
    }
    
    /* Slider */
    .stSlider > div > div > div > div {
        background: #2a3344 !important;
    }
    
    .stSlider > div > div > div > div > div {
        background: linear-gradient(90deg, #4f8cf7, #6d28d9) !important;
    }
    
    /* ── Send button ── */
    .stForm > div:last-child button {
        background: linear-gradient(135deg, #4f8cf7, #6d28d9) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 10px 0 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 12px rgba(79, 140, 247, 0.25) !important;
        width: 100% !important;
    }
    
    .stForm > div:last-child button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(79, 140, 247, 0.35) !important;
    }
    
    .stForm > div:last-child button:active {
        transform: translateY(0);
    }
    
    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 4px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { 
        background: #2a3344;
        border-radius: 100px; 
    }
    ::-webkit-scrollbar-thumb:hover { 
        background: #4f8cf7;
    }
    
    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid #1a1f2e;
        margin: 1.5rem 0;
    }
    
    /* ── Spinner ── */
    .stSpinner > div {
        border-color: #4f8cf7 !important;
        border-top-color: transparent !important;
    }
    
    /* ── Alert boxes ── */
    .stAlert {
        border-radius: 8px !important;
        background: #141922 !important;
        border: 1px solid #2a3344 !important;
    }
    
    /* ── Topic items ── */
    .topic-item {
        padding: 4px 0;
        color: #94a3b8;
        font-size: 0.9rem;
    }
    
    /* ── Sidebar model info ── */
    .model-info {
        padding: 0.75rem 1rem;
        background: #141922;
        border-radius: 8px;
        border: 1px solid #1a1f2e;
        margin-top: 1rem;
    }
    
    .model-info p {
        font-size: 0.7rem;
        color: #64748b !important;
        line-height: 1.8;
        margin: 0;
    }
    
    .model-info strong {
        color: #94a3b8 !important;
    }
    
    /* ── Footer ── */
    .footer {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
        font-size: 0.75rem;
        color: #475569;
        border-top: 1px solid #1a1f2e;
        margin-top: 1.5rem;
    }
    
    /* ── Mobile responsiveness ── */
    @media (max-width: 640px) {
        .main .block-container {
            padding: 1rem !important;
        }
        
        .user-bubble, .assistant-bubble, .emergency-bubble {
            max-width: 95% !important;
        }
        
        .welcome-container {
            padding: 2rem 1rem;
        }
        
        .welcome-title {
            font-size: 1.5rem;
        }
    }
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
def confidence_badge(level: str) -> str:
    mapping = {
        "High":   ("badge-high",   "● High Confidence"),
        "Medium": ("badge-medium", "● Medium Confidence"),
        "Low":    ("badge-low",    "● Low Confidence"),
        "N/A":    ("badge-na",     "● N/A"),
    }
    cls, label = mapping.get(level, ("badge-na", level))
    return f'<span class="badge {cls}">{label}</span>'


def source_tags(sources: list) -> str:
    if not sources:
        return ""
    tags = " ".join(f'<span class="source-tag">📄 {s}</span>' for s in sources)
    return f'<div style="display:flex; gap:6px; flex-wrap:wrap;">{tags}</div>'


# ── Session state ──────────────────────────────────────────────────────────────
if "chatbot" not in st.session_state:
    with st.spinner("Loading MediAssist knowledge base..."):
        try:
            st.session_state.chatbot    = HealthChatbot(temperature=0.3)
            st.session_state.load_error = None
        except Exception as e:
            st.session_state.chatbot    = None
            st.session_state.load_error = str(e)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 MediAssist")
    st.caption("AI-powered Healthcare Assistant")
    st.divider()

    # Temperature slider
    st.markdown("#### ⚙️ Settings")
    temp = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.temperature,
        step=0.1,
        help="Lower = more factual, Higher = more creative",
    )
    if temp != st.session_state.temperature:
        st.session_state.temperature = temp
        if st.session_state.chatbot:
            st.session_state.chatbot.set_temperature(temp)

    st.divider()

    # Topics
    st.markdown("#### 💬 Topics")
    topics = [
        "🤒 Symptoms & diseases",
        "🥗 Nutrition & diet",
        "🏃 Healthy lifestyle",
        "🛡️ Preventive care",
        "🩹 First aid",
        "🧠 Mental health",
        "😴 Sleep & stress",
    ]
    for t in topics:
        st.markdown(f'<div class="topic-item">{t}</div>', unsafe_allow_html=True)

    st.divider()

    # Suggested questions
    st.markdown("#### 💡 Try asking")
    suggestions = [
        "What are the symptoms of diabetes?",
        "How to lower blood pressure naturally?",
        "What foods boost immunity?",
        "How to perform the Heimlich maneuver?",
        "Signs of heat exhaustion?",
        "How much sleep do adults need?",
    ]
    for q in suggestions:
        if st.button(q, key=f"sug_{q[:25]}", use_container_width=True):
            st.session_state.pending_question = q

    st.divider()

    # New conversation
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        if st.session_state.chatbot:
            st.session_state.chatbot.clear_memory()
        st.rerun()

    # Model info
    import os
    from dotenv import load_dotenv
    load_dotenv()
    
    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    
    st.markdown(f"""
    <div class="model-info">
        <p>
            🤖 <strong>LLM:</strong> Groq · {model_name}<br>
            🔍 <strong>Embeddings:</strong> nomic-embed-text<br>
            🗄️ <strong>Vector DB:</strong> ChromaDB<br>
            🔗 <strong>Framework:</strong> LangChain
        </p>
    </div>
    """, unsafe_allow_html=True)


# ── Main area ──────────────────────────────────────────────────────────────────
# Header
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <h1 style="display: inline-block; margin-right: 0.5rem;">🏥</h1>
    <h1 style="display: inline-block;">MediAssist</h1>
    <p style="color: #94a3b8; margin-top: -0.25rem; font-size: 1.05rem;">
        Your AI-powered healthcare information assistant
    </p>
</div>
""", unsafe_allow_html=True)

# Disclaimer with localized emergency number
st.markdown("""
<div class="disclaimer-box">
    ⚠️ <strong>Disclaimer:</strong> Educational purposes only. 
    Not a substitute for professional medical advice. 
    For emergencies, call <span class="emergency-number">🚑 112 (India)</span>
</div>
""", unsafe_allow_html=True)

# Load error
if st.session_state.load_error:
    st.error(
        f"**Setup required:** {st.session_state.load_error}\n\n"
        "Run in terminal: `python ingest.py`"
    )
    st.stop()


# ── Chat history ───────────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div class="welcome-container">
        <div class="welcome-icon">👋</div>
        <div class="welcome-title">Hello! I'm MediAssist</div>
        <div class="welcome-subtitle">
            Ask me about symptoms, nutrition, lifestyle, first aid, or general
            healthcare topics.
        </div>
    </div>
    """, unsafe_allow_html=True)

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="user-bubble">🧑 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )
    else:
        bubble_class = (
            "emergency-bubble"
            if msg.get("category") == "emergency"
            else "assistant-bubble"
        )
        st.markdown(
            f'<div class="{bubble_class}">🤖 {msg["content"]}</div>',
            unsafe_allow_html=True,
        )

        # Metadata
        meta_parts = [
            confidence_badge(msg.get("confidence", "N/A"))
        ]
        sources = source_tags(msg.get("sources", []))
        if sources:
            meta_parts.append(sources)
        
        if meta_parts:
            st.markdown(
                f'<div class="chat-meta">{" ".join(meta_parts)}</div>',
                unsafe_allow_html=True,
            )


# ── Input form ─────────────────────────────────────────────────────────────────
st.divider()

SAFE_ERROR_RESPONSE = {
    "answer":     "I encountered an unexpected issue processing your request. "
                  "Please try again or rephrase your question.",
    "sources":    [],
    "confidence": "N/A",
    "category":   "error",
}

# Handle suggestion button clicks — process directly, no form needed
if "pending_question" in st.session_state:
    query = st.session_state.pop("pending_question")
    st.session_state.messages.append({"role": "user", "content": query})
    with st.spinner("MediAssist is thinking..."):
        try:
            response = st.session_state.chatbot.chat(query)
        except Exception:
            response = SAFE_ERROR_RESPONSE
    st.session_state.messages.append({
        "role":       "assistant",
        "content":    response["answer"],
        "sources":    response["sources"],
        "confidence": response["confidence"],
        "category":   response["category"],
    })
    st.rerun()

default_input = st.session_state.pop("suggested_question", "")

with st.form(key="chat_form", clear_on_submit=True):
    cols = st.columns([4, 1])
    with cols[0]:
        user_input = st.text_input(
            label="",
            value=default_input,
            placeholder="Ask a healthcare question...",
            label_visibility="collapsed",
        )
    with cols[1]:
        submitted = st.form_submit_button("Send →", use_container_width=True)


# ── Process submission ─────────────────────────────────────────────────────────
if submitted and user_input.strip():
    query = user_input.strip()
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("MediAssist is thinking..."):
        try:
            response = st.session_state.chatbot.chat(query)
        except Exception:
            response = SAFE_ERROR_RESPONSE

    st.session_state.messages.append({
        "role":       "assistant",
        "content":    response["answer"],
        "sources":    response["sources"],
        "confidence": response["confidence"],
        "category":   response["category"],
    })

    st.rerun()


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    MediAssist v1.0 · For informational purposes only
</div>
""", unsafe_allow_html=True)