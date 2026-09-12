import time
import os
import json
from typing import Dict, Any, List, Optional

from src.preprocessor import TextPreprocessor
from src.intent_classifier import AeroGuardIntentClassifier
from src.retriever import DeltaHistoricalRetriever
from src.triage_engine import AeroGuardTriageEngine
from src.generator import AeroGuardReplyGenerator

class AeroGuardAgent:
    """
    AeroGuard: Production AI Customer Support Agent for Delta Air Lines (@Delta).
    End-to-end pipeline:
    1. Preprocessing & PII Redaction
    2. Operational Intent Classification
    3. Grounded Historical Resolution Retrieval (RAG)
    4. Multi-Factor Risk & Escalation Triage
    5. Policy-Grounded Draft Reply Generation
    """
    def __init__(
        self,
        retriever: Optional[DeltaHistoricalRetriever] = None,
        intent_classifier: Optional[AeroGuardIntentClassifier] = None,
        triage_engine: Optional[AeroGuardTriageEngine] = None,
        reply_generator: Optional[AeroGuardReplyGenerator] = None
    ):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        self.preprocessor = TextPreprocessor()
        self.retriever = retriever or DeltaHistoricalRetriever()
        
        # Fit or load intent classifier
        if intent_classifier is None:
            golden_path = os.path.join(base_dir, "data", "golden_eval_set.json")
            with open(golden_path, "r", encoding="utf-8") as f:
                golden_data = json.load(f)
            train_texts = [item["customer_text"] for item in golden_data]
            train_labels = [item["ground_truth_intent"] for item in golden_data]
            self.intent_classifier = AeroGuardIntentClassifier(confidence_threshold=0.25).fit(train_texts, train_labels)
        else:
            self.intent_classifier = intent_classifier
            
        self.triage_engine = triage_engine or AeroGuardTriageEngine()
        self.generator = reply_generator or AeroGuardReplyGenerator()

    def process_tweet(self, tweet_text: str, top_k_rag: int = 3) -> Dict[str, Any]:
        """
        Executes end-to-end customer support workflow for an incoming tweet.
        """
        start_time = time.time()

        # Step 1: Preprocessing & PII Detection
        pii_result = self.preprocessor.sanitize_pii(tweet_text)
        signals = self.preprocessor.extract_signals(tweet_text)

        # Step 2: Intent Classification
        intent_res = self.intent_classifier.predict(pii_result["sanitized_text"])
        intent = intent_res["intent"]
        intent_conf = intent_res["confidence"]

        # Step 3: Grounded Resolution Retrieval (RAG)
        retrieved_resolutions = self.retriever.retrieve(pii_result["sanitized_text"], top_k=top_k_rag)

        # Step 4: Multi-Factor Risk & Escalation Triage
        triage_res = self.triage_engine.decide(
            text=pii_result["cleaned_text"],
            intent=intent,
            intent_confidence=intent_conf,
            pii_info=pii_result,
            signals=signals
        )

        # Step 5: Grounded Reply Generation
        reply_res = self.generator.generate_reply(
            customer_text=pii_result["cleaned_text"],
            intent=intent,
            triage_decision=triage_res,
            retrieved_contexts=retrieved_resolutions,
            pii_info=pii_result
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "original_tweet": tweet_text,
            "sanitized_tweet": pii_result["sanitized_text"],
            "pii_detected": pii_result["pii_detected"],
            "pii_entities": pii_result["entities"],
            "intent": intent,
            "intent_confidence": intent_conf,
            "intent_probabilities": intent_res.get("probabilities", {}),
            "is_out_of_distribution": intent_res.get("is_out_of_distribution", False),
            "signals": signals,
            "triage_decision": triage_res["decision"],
            "should_escalate": triage_res["should_escalate"],
            "escalation_reason": triage_res["reason"],
            "triage_confidence": triage_res["confidence"],
            "risk_score": triage_res["risk_score"],
            "risk_factors": triage_res["risk_factors"],
            "draft_reply": reply_res["reply"],
            "retrieved_resolutions": retrieved_resolutions,
            "execution_latency_ms": latency_ms
        }
