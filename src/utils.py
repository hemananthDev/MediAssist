"""
src/utils.py
------------
Guardrail checks and shared utility functions.
"""

# ── Keyword lists ──────────────────────────────────────────────────────────────

EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "difficulty breathing", "not breathing", "unconscious", "unresponsive",
    "severe bleeding", "overdose", "poisoning", "suicid", "choking",
]

# Phrases that indicate a user wants a personal diagnosis
DIAGNOSIS_KEYWORDS = [
    "do i have", "have i got", "is it cancer", "is this cancer",
    "am i diabetic", "do i have diabetes", "what disease do i have",
    "diagnose me", "what illness do i have", "what condition do i have",
]

# Phrases about changing or stopping medication
MEDICATION_KEYWORDS = [
    "should i stop", "can i stop", "stop taking my", "stop my medication",
    "stop my medicine", "change my medication", "change my medicine",
    "should i take", "can i take", "should i increase", "should i decrease",
]

# Clearly non-healthcare topics
OFF_TOPIC_KEYWORDS = [
    "stock market", "cryptocurrency", "bitcoin", "ethereum",
    "election", "politics", "political", "president",
    "movie", "film", "netflix", "music", "song", "album",
    "weather forecast", "sports score", "football", "cricket score",
    "programming", "write code", "software bug",
]

# Harmful / safety-rejection topics
SAFETY_KEYWORDS = [
    "bomb", "weapon", "explosive", "hack", "poison someone",
    "how to kill", "hurt someone",
]


# ── Guardrail functions ────────────────────────────────────────────────────────

def is_emergency(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in EMERGENCY_KEYWORDS)


def is_diagnosis_request(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in DIAGNOSIS_KEYWORDS)


def is_medication_change_request(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in MEDICATION_KEYWORDS)


def is_off_topic(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in OFF_TOPIC_KEYWORDS)


def is_safety_violation(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in SAFETY_KEYWORDS)


def is_low_confidence_answer(answer: str) -> bool:
    """
    Detect if the LLM itself indicated it couldn't find an answer,
    so the UI can show the fallback confidence state.
    """
    low_confidence_phrases = [
        "i couldn't find",
        "not in the context",
        "i don't have information",
        "i'm not sure",
        "no information available",
    ]
    a = answer.lower()
    return any(phrase in a for phrase in low_confidence_phrases)
