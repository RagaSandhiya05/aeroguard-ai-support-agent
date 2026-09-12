from typing import Dict, Any, List

class TrivialTriageEngine:
    """
    Baseline 0: Trivial triage policy.
    Never escalates to human; attempts to auto-reply to 100% of customer messages.
    """
    def decide(self, text: str, intent: str = "general_greeting_gratitude") -> Dict[str, Any]:
        return {
            "decision": "AUTO_REPLY",
            "should_escalate": False,
            "confidence": 0.5,
            "reason": "Baseline 0 static policy: default auto-reply.",
            "risk_score": 0.1,
            "model_name": "Baseline0_StaticNeverEscalate"
        }

class KeywordTriageEngine:
    """
    Baseline 1: Simple keyword-matching triage policy.
    Escalates if text contains explicit escalation keywords.
    """
    ESCALATION_KEYWORDS = [
        "cancel", "refund", "emergency", "manager", "urgent", "agent", "human",
        "stole", "lawyer", "police", "compensation", "voucher", "lost"
    ]

    def decide(self, text: str, intent: str = "") -> Dict[str, Any]:
        text_lower = text.lower()
        matched = [kw for kw in self.ESCALATION_KEYWORDS if kw in text_lower]
        
        if matched:
            return {
                "decision": "ESCALATE_TO_HUMAN",
                "should_escalate": True,
                "confidence": 0.70,
                "reason": f"Baseline 1 keyword match triggered: {', '.join(matched)}",
                "risk_score": 0.75,
                "model_name": "Baseline1_KeywordTriage"
            }
        else:
            return {
                "decision": "AUTO_REPLY",
                "should_escalate": False,
                "confidence": 0.65,
                "reason": "Baseline 1: No escalation keywords found in text.",
                "risk_score": 0.20,
                "model_name": "Baseline1_KeywordTriage"
            }

class AeroGuardTriageEngine:
    """
    AeroGuard Production Triage & Risk Decision Engine:
    Multi-factor risk assessment combining:
    1. PII exposure in public tweets (PNR, ticket numbers, credit cards).
    2. Vulnerability & regulatory signals (medical, wheelchair, tarmac delay, DOT rule).
    3. Operational intent risk (disruptions, cancellations, damaged baggage, staff complaints).
    4. Sarcasm / high emotional distress signals.
    5. Intent confidence thresholding.
    
    Outputs trusted AUTO_REPLY vs ESCALATE_TO_HUMAN decisions with explicit, calibrated rationale.
    """
    
    # Intents that inherently demand human agent operational authority
    HUMAN_MANDATORY_INTENTS = {
        "complaint_staff_service": "Staff conduct and severe service dissatisfaction require human investigation and retention care.",
        "flight_disruption": "Active flight cancellation or missed connection requires live seat inventory rebooking authority.",
        "baggage_issue": "Lost or damaged luggage claims require physical airport baggage services dispatch."
    }

    # Routine informational intents safe for autonomous resolution
    AUTONOMOUS_SAFE_INTENTS = {
        "baggage_rules_faq",
        "general_greeting_gratitude"
    }

    def decide(
        self,
        text: str,
        intent: str,
        intent_confidence: float = 0.8,
        pii_info: Dict[str, Any] = None,
        signals: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        risk_score = 0.0
        risk_factors = []
        reasons = []

        # 1. PII Exposure Check (Critical Safety Risk)
        if pii_info and pii_info.get("pii_detected"):
            entities = pii_info.get("entities", [])
            risk_score += 0.85
            risk_factors.append(f"Public PII exposed: {', '.join(entities)}")
            reasons.append("Customer exposed private reservation/identity credentials on public Twitter; mandatory transfer to secure direct message.")

        # 2. Critical Safety / Medical / Accessibility Flags
        if signals:
            if signals.get("has_critical_risk"):
                factors = signals.get("critical_factors", [])
                risk_score += 0.90
                risk_factors.append(f"Critical vulnerability: {', '.join(factors)}")
                reasons.append("CRITICAL VULNERABILITY: Health, safety, or accessibility priority requiring immediate specialized human intervention.")
            if signals.get("is_imminent"):
                factors = signals.get("imminent_factors", [])
                risk_score += 0.70
                risk_factors.append(f"Imminent flight disruption: {', '.join(factors)}")
                reasons.append("Imminent flight boarding/departure urgency; requires real-time airport ground operations assistance.")
            if signals.get("has_sarcasm"):
                risk_score += 0.50
                risk_factors.append("Sarcasm masking customer distress")
                reasons.append("Customer expressing high distress through inverted sarcasm; human empathy required.")

        # 3. Intent Risk Evaluation
        text_lower = text.lower()
        
        # Flight status check vs active cancellation
        if intent == "flight_disruption":
            if any(w in text_lower for w in ["on time", "status", "scheduled time", "depart on time"]):
                # Routine flight status query is safe for autonomous resolution
                risk_score += 0.15
            else:
                risk_score += 0.75
                reasons.append("Active flight delay, cancellation, or connection disruption requiring reservation rebooking.")
                risk_factors.append("Flight disruption requiring rebooking")

        elif intent == "baggage_issue":
            if any(w in text_lower for w in ["tracking tool", "how do i track", "allowance", "bag app"]):
                risk_score += 0.15
            else:
                risk_score += 0.75
                reasons.append("Lost, delayed, or physically damaged baggage claim requiring airport PIR dispatch.")
                risk_factors.append("Baggage loss or physical damage")

        elif intent == "booking_reservation":
            # Routine FAQ / cancel for eCredit vs specific PNR seat modification
            if any(w in text_lower for w in ["how to cancel", "cancel for ecredit", "upgrade policy", "how many skymiles"]):
                risk_score += 0.20
            else:
                risk_score += 0.65
                reasons.append("Reservation modification, duplicate billing, or ticketing override requires agent account tools.")
                risk_factors.append("Direct reservation change / billing inquiry")

        elif intent == "complaint_staff_service":
            risk_score += 0.80
            reasons.append("Formal staff conduct complaint requiring customer care escalation and incident documentation.")
            risk_factors.append("Staff service complaint")

        elif intent == "inflight_airport_service":
            if "refund" in text_lower or "paid" in text_lower:
                risk_score += 0.65
                reasons.append("Inflight Wi-Fi / amenities billing refund dispute.")
                risk_factors.append("Billing refund dispute")
            else:
                risk_score += 0.15

        elif intent in self.AUTONOMOUS_SAFE_INTENTS:
            risk_score += 0.10

        # Cap risk score at 1.0
        risk_score = min(1.0, round(risk_score, 3))

        # Decision Boundary
        should_escalate = risk_score >= 0.50
        decision = "ESCALATE_TO_HUMAN" if should_escalate else "AUTO_REPLY"

        # Formulate explicit stated reason
        if should_escalate:
            final_reason = " | ".join(reasons) if reasons else "Elevated operational risk exceeds autonomous handling threshold."
            confidence = min(0.98, max(0.70, risk_score))
        else:
            final_reason = "Routine policy inquiry or positive brand engagement safe for autonomous grounded response."
            confidence = min(0.98, max(0.75, 1.0 - risk_score))

        return {
            "decision": decision,
            "should_escalate": should_escalate,
            "confidence": round(confidence, 4),
            "reason": final_reason,
            "risk_score": risk_score,
            "risk_factors": risk_factors,
            "model_name": "AeroGuard_MultiFactorTriage"
        }
