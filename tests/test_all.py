import pytest
import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

from src.preprocessor import TextPreprocessor
from src.intent_classifier import TrivialIntentClassifier, SimpleIntentClassifier, AeroGuardIntentClassifier
from src.retriever import DeltaHistoricalRetriever
from src.triage_engine import TrivialTriageEngine, KeywordTriageEngine, AeroGuardTriageEngine
from src.generator import TrivialReplyGenerator, AeroGuardReplyGenerator
from src.agent import AeroGuardAgent
from src.judge import RubricEvaluator

@pytest.fixture(scope="module")
def preprocessor():
    return TextPreprocessor()

@pytest.fixture(scope="module")
def retriever():
    return DeltaHistoricalRetriever()

@pytest.fixture(scope="module")
def agent(retriever):
    return AeroGuardAgent(retriever=retriever)

def test_pnr_redaction(preprocessor):
    text = "My booking code is H7K9P2. Please check my seat."
    res = preprocessor.sanitize_pii(text)
    assert res["pii_detected"] is True
    assert "PNR_CONFIRMATION_CODE" in res["entities"]
    assert "H7K9P2" not in res["sanitized_text"]
    assert "[REDACTED_PNR]" in res["sanitized_text"]

def test_ticket_number_redaction(preprocessor):
    text = "Double charged on ticket 0061234567890."
    res = preprocessor.sanitize_pii(text)
    assert res["pii_detected"] is True
    assert "TICKET_NUMBER" in res["entities"]
    assert "[REDACTED_TICKET_NUMBER]" in res["sanitized_text"]

def test_signal_extraction(preprocessor):
    text = "My wheelchair was broken and we are stuck on tarmac for 3 hours!"
    sig = preprocessor.extract_signals(text)
    assert sig["has_critical_risk"] is True
    assert "wheelchair" in sig["critical_factors"]

def test_intent_classifiers():
    train_x = ["flight cancelled", "baggage lost on carousel", "how to cancel for ecredit", "great service thanks"]
    train_y = ["flight_disruption", "baggage_issue", "booking_reservation", "general_greeting_gratitude"]
    
    ag = AeroGuardIntentClassifier(confidence_threshold=0.2).fit(train_x, train_y)
    pred = ag.predict("My suitcase was lost on the carousel")
    assert pred["intent"] == "baggage_issue"
    assert pred["confidence"] > 0.3

def test_triage_pii_mandatory_escalation():
    triage = AeroGuardTriageEngine()
    pii_info = {"pii_detected": True, "entities": ["PNR_CONFIRMATION_CODE"]}
    decision = triage.decide(text="Here is my code H7K9P2", intent="booking_reservation", pii_info=pii_info)
    assert decision["decision"] == "ESCALATE_TO_HUMAN"
    assert decision["should_escalate"] is True
    assert "private" in decision["reason"].lower() or "credentials" in decision["reason"].lower()

def test_triage_routine_faq_auto_reply():
    triage = AeroGuardTriageEngine()
    pii_info = {"pii_detected": False}
    decision = triage.decide(text="What are the carry on bag dimensions?", intent="baggage_rules_faq", pii_info=pii_info)
    assert decision["decision"] == "AUTO_REPLY"
    assert decision["should_escalate"] is False

def test_end_to_end_agent_pipeline(agent):
    res = agent.process_tweet("What is the pet policy for traveling in cabin on Delta?")
    assert res["intent"] == "baggage_rules_faq"
    assert res["triage_decision"] == "AUTO_REPLY"
    assert "delta.com" in res["draft_reply"].lower() or "pet" in res["draft_reply"].lower()
    assert "*AI" in res["draft_reply"]
    assert res["execution_latency_ms"] < 200

def test_judge_rubric_evaluation():
    judge = RubricEvaluator()
    score = judge.evaluate_reply(
        customer_text="What are carry-on dimensions?",
        generated_reply="Delta carry-on bags must meet 22x14x9 inches. Visit delta.com/baggage. *AI",
        reference_reply="Standard carry-on size is 22x14x9. *SK",
        intent="baggage_rules_faq",
        pii_detected=False,
        triage_decision="AUTO_REPLY"
    )
    assert score["overall_score"] >= 4.0
    assert score["categorical_rating"] == "EXCELLENT"
