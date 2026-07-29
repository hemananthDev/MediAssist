"""
src/utils.py
------------
Guardrail checks and shared utility functions.

Design principle: prefer false negatives over false positives.
It's better to let a borderline query reach the LLM (which has its own
system-prompt rules) than to wrongly block a legitimate healthcare question.
"""

import re

# ── Emergency keywords (life-threatening — these must stay broad) ──────────────
EMERGENCY_KEYWORDS = [
    "chest pain", "heart attack", "stroke", "can't breathe", "cannot breathe",
    "difficulty breathing", "not breathing", "unconscious", "unresponsive",
    "severe bleeding", "overdose", "poisoning", "suicid", "choking",
]

# ── Diagnosis keywords — only fire on explicit personal diagnosis requests ─────
# Kept specific to avoid blocking nutrition/lifestyle questions that use
# similar phrasing (e.g. "do i have to avoid dairy", "do i need vitamins")
DIAGNOSIS_PATTERNS = [
    r"\bdo i have\s+(cancer|diabetes|hiv|aids|tb|tuberculosis|covid|flu|malaria|dengue|typhoid|asthma|arthritis|depression|anxiety|hypertension|a (disease|condition|illness|disorder))\b",
    r"\bhave i got\s+(cancer|diabetes|hiv|aids|a (disease|condition|illness))\b",
    r"\bdiagnose me\b",
    r"\bwhat (disease|illness|condition|disorder) do i have\b",
    r"\bam i (diabetic|hypertensive|infected|sick with)\b",
    r"\bis (it|this) (cancer|diabetes|hiv|a tumor|terminal)\b",
]

# ── Medication change keywords — only fire on explicit change/stop requests ────
# Avoids blocking: "what vitamins should I take", "can I take ginger for nausea"
# Only blocks: "should I stop my medication", "can I stop taking my pills"
MEDICATION_PATTERNS = [
    r"\bstop (taking |my )(my )?(\w+\s+){0,3}(medication|medicine|pills|tablets|drugs|prescription|dose|dosage)\b",
    r"\bchange (my )?(\w+\s+){0,3}(medication|medicine|prescription)\b",
    r"\bshould i (stop|quit|discontinue|reduce|increase|double) (taking )?(my )?(\w+\s+){0,3}(medication|medicine|pills|tablets|prescription|dose|dosage)\b",
    r"\bcan i (stop|quit|discontinue) (taking )?(my )?(\w+\s+){0,3}(medication|medicine|pills|tablets)\b",
    r"\bshould i (increase|decrease|reduce|adjust) (my )?(\w+\s+){0,3}(dose|dosage)\b",
]

# ── Off-topic keywords — only clearly non-healthcare domains ──────────────────
# Removed: "football", "sports score", "film" — injury/health questions
# often mention these contexts. Kept only unambiguous non-health topics.
OFF_TOPIC_KEYWORDS = [
    "stock market", "stock price", "share price",
    "cryptocurrency", "bitcoin", "ethereum", "crypto trading",
    "election result", "election campaign", "who won the election",
    "movie review", "film review", "review this movie", "netflix show", "tv series",
    "music album", "song lyrics",
    "weather forecast", "weather tomorrow",
    "write code", "debug code", "debug my", "software bug", "programming tutorial",
]

# ── Safety keywords — harmful intent ─────────────────────────────────────────
SAFETY_KEYWORDS = [
    "how to make a bomb", "build a weapon", "make explosives",
    "poison someone", "how to kill someone", "how to hurt someone",
    "hack into", "create malware",
]

# ── Greeting keywords — short social exchanges ────────────────────────────────
GREETING_PATTERNS = [
    r"^(hi|hello|hey|howdy|greetings|good (morning|afternoon|evening|night))[\s!?.]*$",
    r"^how are you[\s!?.]*$",
    r"^what('s| is) up[\s!?.]*$",
    r"^(thanks|thank you|ty|thx)[\s!?.]*$",
    r"^(bye|goodbye|see you|cya)[\s!?.]*$",
]


# ── Guardrail functions ────────────────────────────────────────────────────────

def is_emergency(query: str) -> bool:
    """Broad match — life safety takes priority over false positives."""
    q = query.lower()
    return any(kw in q for kw in EMERGENCY_KEYWORDS)


def is_diagnosis_request(query: str) -> bool:
    """
    Only fires on explicit personal diagnosis requests for named conditions.
    General questions like 'do i have to avoid dairy' are NOT blocked.
    """
    q = query.lower()
    return any(re.search(pattern, q) for pattern in DIAGNOSIS_PATTERNS)


def is_medication_change_request(query: str) -> bool:
    """
    Only fires when user explicitly asks to stop/change/adjust a medication.
    Questions like 'what vitamins should I take' are NOT blocked.
    """
    q = query.lower()
    return any(re.search(pattern, q) for pattern in MEDICATION_PATTERNS)


def is_off_topic(query: str) -> bool:
    """
    Only fires on clearly non-healthcare domains.
    Sports/activity injuries, film-related health questions are NOT blocked.
    """
    q = query.lower()
    return any(kw in q for kw in OFF_TOPIC_KEYWORDS)


def is_safety_violation(query: str) -> bool:
    """Fires only on explicit harmful-intent phrases."""
    q = query.lower()
    return any(kw in q for kw in SAFETY_KEYWORDS)


def is_greeting(query: str) -> bool:
    """Detect short social greetings that don't need RAG retrieval."""
    q = query.strip().lower()
    return any(re.search(pattern, q) for pattern in GREETING_PATTERNS)


def is_low_confidence_answer(answer: str) -> bool:
    """
    Detect if the LLM indicated it couldn't find an answer.
    Only checks the first 15 words to avoid discarding nuanced responses
    that happen to contain hedging language mid-sentence.
    """
    first_words = " ".join(answer.strip().split()[:15]).lower()
    low_confidence_phrases = [
        "i couldn't find",
        "i could not find",
        "not in the context",
        "i don't have information",
        "no information available",
        "i was unable to find",
    ]
    return any(phrase in first_words for phrase in low_confidence_phrases)