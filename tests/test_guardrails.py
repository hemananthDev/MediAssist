"""
tests/test_guardrails.py
------------------------
Unit tests for all guardrail functions in src/utils.py.

Covers:
  - True positives  : phrases that MUST be caught
  - False positives : legitimate healthcare phrases that MUST NOT be blocked
  - Edge cases      : casing, punctuation, mixed phrasing

Run with:
    pytest tests/ -v
"""

import pytest
from src.utils import (
    is_emergency,
    is_diagnosis_request,
    is_medication_change_request,
    is_off_topic,
    is_safety_violation,
    is_greeting,
    is_low_confidence_answer,
)


# ══════════════════════════════════════════════════════════════════════════════
# is_emergency
# ══════════════════════════════════════════════════════════════════════════════

class TestIsEmergency:

    # Must catch
    @pytest.mark.parametrize("query", [
        "I have chest pain",
        "He is having a heart attack",
        "She had a stroke",
        "I can't breathe",
        "I cannot breathe",
        "The patient is unconscious",
        "There is severe bleeding",
        "I think I took an overdose",
        "He is choking on food",
        "I am having difficulty breathing",
        "suicidal thoughts",
    ])
    def test_catches_emergencies(self, query):
        assert is_emergency(query) is True, f"Should catch: {query!r}"

    # Must NOT catch
    @pytest.mark.parametrize("query", [
        "What are the symptoms of diabetes?",
        "How do I eat healthy?",
        "What is the DASH diet?",
        "How much water should I drink?",
        "Tell me about asthma",
        "What is a normal blood pressure?",
    ])
    def test_allows_normal_queries(self, query):
        assert is_emergency(query) is False, f"Should allow: {query!r}"


# ══════════════════════════════════════════════════════════════════════════════
# is_diagnosis_request
# ══════════════════════════════════════════════════════════════════════════════

class TestIsDiagnosisRequest:

    # Must catch
    @pytest.mark.parametrize("query", [
        "Do I have cancer?",
        "Do I have diabetes?",
        "Do I have hypertension?",
        "Diagnose me",
        "What disease do I have?",
        "What illness do I have?",
        "Am I diabetic?",
        "Is this cancer?",
        "Have I got HIV?",
        "What condition do I have?",
    ])
    def test_catches_diagnosis_requests(self, query):
        assert is_diagnosis_request(query) is True, f"Should catch: {query!r}"

    # Must NOT catch — these are legitimate healthcare questions
    @pytest.mark.parametrize("query", [
        "Do I have to avoid dairy if I'm lactose intolerant?",
        "Do I need to take vitamin D supplements?",
        "Do I have to exercise every day?",
        "What foods should I eat if I have high blood pressure?",
        "What are the symptoms of diabetes?",
        "How is diabetes diagnosed?",
        "Can diabetes be reversed?",
        "What vitamins should I take for immunity?",
        "How do I know if I have the flu?",
    ])
    def test_allows_legitimate_healthcare_queries(self, query):
        assert is_diagnosis_request(query) is False, f"Should allow: {query!r}"


# ══════════════════════════════════════════════════════════════════════════════
# is_medication_change_request
# ══════════════════════════════════════════════════════════════════════════════

class TestIsMedicationChangeRequest:

    # Must catch
    @pytest.mark.parametrize("query", [
        "Should I stop my medication?",
        "Can I stop taking my pills?",
        "Should I stop taking my blood pressure medication?",
        "Should I stop taking my diabetes medicine?",
        "Can I stop taking my heart medication?",
        "Should I change my prescription?",
        "Should I reduce my dose?",
        "Should I increase my dosage?",
        "Stop taking my medication",
        "Can I discontinue my tablets?",
        "Should I quit taking my medicine?",
    ])
    def test_catches_medication_change_requests(self, query):
        assert is_medication_change_request(query) is True, f"Should catch: {query!r}"

    # Must NOT catch — the original false positive cases
    @pytest.mark.parametrize("query", [
        "What vitamins should I take for immunity?",
        "Can I take ginger for nausea?",
        "Should I take vitamin C when I have a cold?",
        "Can I take paracetamol for a headache?",
        "What supplements should I take?",
        "Should I take iron with vitamin C?",
        "Can I take probiotics daily?",
        "What should I take for a fever?",
        "Should I take more magnesium?",
    ])
    def test_allows_supplement_and_general_queries(self, query):
        assert is_medication_change_request(query) is False, f"Should allow: {query!r}"


# ══════════════════════════════════════════════════════════════════════════════
# is_off_topic
# ══════════════════════════════════════════════════════════════════════════════

class TestIsOffTopic:

    # Must catch
    @pytest.mark.parametrize("query", [
        "What is the stock price of Apple?",
        "Tell me about Bitcoin",
        "What is the weather forecast for tomorrow?",
        "Who won the election?",
        "Write code for a web scraper",
        "Review this movie for me",
        "What is the latest music album?",
        "Debug my Python code",
    ])
    def test_catches_off_topic(self, query):
        assert is_off_topic(query) is True, f"Should catch: {query!r}"

    # Must NOT catch — the original false positive cases
    @pytest.mark.parametrize("query", [
        "I hurt my ankle playing football, what first aid should I do?",
        "What should I eat after a sports injury?",
        "I watched a film about cancer, is smoking really that bad?",
        "Is running good for heart health?",
        "What are the health risks of alcohol?",
        "How do I treat a sports injury?",
        "What is the best diet for athletes?",
    ])
    def test_allows_health_adjacent_queries(self, query):
        assert is_off_topic(query) is False, f"Should allow: {query!r}"


# ══════════════════════════════════════════════════════════════════════════════
# is_safety_violation
# ══════════════════════════════════════════════════════════════════════════════

class TestIsSafetyViolation:

    @pytest.mark.parametrize("query", [
        "How to make a bomb",
        "How to kill someone",
        "How to hurt someone",
        "Poison someone with household chemicals",
        "hack into a hospital system",
    ])
    def test_catches_safety_violations(self, query):
        assert is_safety_violation(query) is True, f"Should catch: {query!r}"

    @pytest.mark.parametrize("query", [
        "What foods are good for heart health?",
        "How do I treat a poisoning victim?",
        "What is the antidote for paracetamol overdose?",
        "How do I stay safe in the heat?",
    ])
    def test_allows_safe_healthcare_queries(self, query):
        assert is_safety_violation(query) is False, f"Should allow: {query!r}"


# ══════════════════════════════════════════════════════════════════════════════
# is_greeting
# ══════════════════════════════════════════════════════════════════════════════

class TestIsGreeting:

    @pytest.mark.parametrize("query", [
        "hi",
        "Hi!",
        "hello",
        "Hello!",
        "hey",
        "good morning",
        "Good Evening",
        "how are you",
        "How are you?",
        "thanks",
        "Thank you!",
        "bye",
        "Goodbye",
    ])
    def test_catches_greetings(self, query):
        assert is_greeting(query) is True, f"Should catch: {query!r}"

    @pytest.mark.parametrize("query", [
        "hi, what are the symptoms of diabetes?",
        "hello doctor, I have a fever",
        "good morning, can you help me with nutrition advice?",
        "What is hypertension?",
        "How do I lose weight?",
    ])
    def test_allows_greetings_with_health_content(self, query):
        assert is_greeting(query) is False, f"Should allow: {query!r}"


# ══════════════════════════════════════════════════════════════════════════════
# is_low_confidence_answer
# ══════════════════════════════════════════════════════════════════════════════

class TestIsLowConfidenceAnswer:

    @pytest.mark.parametrize("answer", [
        "I couldn't find reliable information in my knowledge base for that question.",
        "I could not find any relevant context for this query.",
        "I was unable to find information on this topic.",
        "No information available on this subject in the knowledge base.",
    ])
    def test_catches_low_confidence(self, answer):
        assert is_low_confidence_answer(answer) is True, f"Should catch: {answer[:50]!r}"

    @pytest.mark.parametrize("answer", [
        # Good answer that happens to use hedging mid-sentence — must NOT be discarded
        "Hypertension is a condition where blood pressure is elevated. I'm not sure how it compares to other cardiovascular conditions, but generally a reading above 130/80 mmHg is considered high.",
        # Normal confident answer
        "The symptoms of diabetes include frequent urination, excessive thirst, and unexplained weight loss.",
        # Short answer
        "Drink at least 8 glasses of water per day.",
        # Answer with 'not sure' but not at start
        "Exercise is generally beneficial. I'm not sure of the exact mechanism, but it reduces blood pressure.",
    ])
    def test_allows_good_answers_with_hedging(self, answer):
        assert is_low_confidence_answer(answer) is False, f"Should allow: {answer[:60]!r}"
