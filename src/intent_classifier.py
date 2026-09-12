import json
import os
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from collections import Counter

TAXONOMY = [
    "flight_disruption",
    "baggage_issue",
    "booking_reservation",
    "baggage_rules_faq",
    "inflight_airport_service",
    "complaint_staff_service",
    "general_greeting_gratitude"
]

class TrivialIntentClassifier:
    """
    Baseline 0: Trivial majority-class classifier.
    Always predicts the most frequent class in historical customer support interactions.
    """
    def __init__(self, majority_class: str = "flight_disruption"):
        self.majority_class = majority_class

    def fit(self, X: List[str], y: List[str]):
        counts = Counter(y)
        self.majority_class = counts.most_common(1)[0][0]
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        return {
            "intent": self.majority_class,
            "confidence": 0.5,
            "probabilities": {cls: (1.0 if cls == self.majority_class else 0.0) for cls in TAXONOMY},
            "model_name": "Baseline0_MajorityClass"
        }

class SimpleIntentClassifier:
    """
    Baseline 1: Standard TF-IDF + Logistic Regression classifier.
    Uses default word n-grams (1, 1).
    """
    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 1), lowercase=True, max_features=2500)),
            ("clf", LogisticRegression(max_iter=300, random_state=42))
        ])
        self.is_fitted = False

    def fit(self, X: List[str], y: List[str]):
        self.pipeline.fit(X, y)
        self.classes_ = list(self.pipeline.classes_)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("Classifier is not fitted.")
        probs = self.pipeline.predict_proba([text])[0]
        top_idx = int(np.argmax(probs))
        pred_intent = self.classes_[top_idx]
        confidence = float(probs[top_idx])
        
        prob_dict = {cls: float(p) for cls, p in zip(self.classes_, probs)}
        return {
            "intent": pred_intent,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict,
            "model_name": "Baseline1_TfidfLogReg"
        }

class AeroGuardIntentClassifier:
    """
    AeroGuard Production Intent Classifier:
    - FeatureUnion of word n-grams (1, 3) and character n-grams (3, 5) for typo and Twitter slang tolerance.
    - Class-weight balanced multinomial logistic regression with L2 regularization.
    - Calibrated confidence thresholding with Out-Of-Distribution (OOD) detection.
    - Lexical domain rule boost for critical aviation safety and policy tokens.
    """
    def __init__(self, confidence_threshold: float = 0.35):
        self.confidence_threshold = confidence_threshold
        
        # Word + Char n-gram union for high robustness against typos ("cancellled", "rebookin")
        self.feature_extractor = FeatureUnion([
            ("word_tfidf", TfidfVectorizer(ngram_range=(1, 3), sublinear_tf=True, min_df=1, max_features=5000)),
            ("char_tfidf", TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=1, max_features=5000))
        ])
        self.clf = LogisticRegression(C=2.5, class_weight="balanced", max_iter=500, random_state=42)
        self.is_fitted = False
        self.classes_ = []

    def fit(self, X: List[str], y: List[str]):
        features = self.feature_extractor.fit_transform(X)
        self.clf.fit(features, y)
        self.classes_ = list(self.clf.classes_)
        self.is_fitted = True
        return self

    def predict(self, text: str) -> Dict[str, Any]:
        if not self.is_fitted:
            raise ValueError("AeroGuard Intent Classifier is not fitted.")
            
        features = self.feature_extractor.transform([text])
        probs = self.clf.predict_proba(features)[0]
        prob_dict = {cls: float(p) for cls, p in zip(self.classes_, probs)}
        
        top_idx = int(np.argmax(probs))
        top_intent = self.classes_[top_idx]
        confidence = float(probs[top_idx])
        
        # Domain rule override for distinctive unambiguous keywords
        text_lower = text.lower()
        if any(w in text_lower for w in ["carousel", "lost bag", "delayed bag", "damaged luggage", "missing bag", "broken wheel"]):
            if "baggage_issue" in prob_dict and prob_dict["baggage_issue"] > 0.2:
                top_intent = "baggage_issue"
                confidence = max(confidence, 0.85)
        elif any(w in text_lower for w in ["carry on dimensions", "baggage allowance", "pet in cabin", "golf bag fee", "checked bag fee"]):
            if "baggage_rules_faq" in prob_dict and prob_dict["baggage_rules_faq"] > 0.2:
                top_intent = "baggage_rules_faq"
                confidence = max(confidence, 0.88)
        elif any(w in text_lower for w in ["tarmac", "divert", "cancelled flight", "flight cancelled", "delay"]):
            if "flight_disruption" in prob_dict and prob_dict["flight_disruption"] > 0.2:
                top_intent = "flight_disruption"
                confidence = max(confidence, 0.86)

        # OOD fallback
        is_ood = confidence < self.confidence_threshold
        final_intent = "uncertain_dispatched" if is_ood else top_intent
        
        return {
            "intent": final_intent,
            "raw_intent": top_intent,
            "confidence": round(confidence, 4),
            "is_out_of_distribution": is_ood,
            "probabilities": {k: round(v, 4) for k, v in sorted(prob_dict.items(), key=lambda item: -item[1])},
            "model_name": "AeroGuard_HybridIntentClassifier"
        }
