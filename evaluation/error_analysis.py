import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import json
from src.agent import AeroGuardAgent

def analyze_top_failures():
    agent = AeroGuardAgent()
    
    failure_cases = [
        {
            "failure_id": "FAIL_01",
            "category": "Sarcasm & Sentiment Inversion",
            "customer_tweet": "Thanks so much Delta for another magical 6 hours sitting on the floor at Atlanta airport! You guys never fail to impress!",
            "ground_truth_intent": "flight_disruption",
            "ground_truth_escalate": True,
            "root_cause_hypothesis": "Lexical positive sentiment tokens ('magical', 'impress', 'thanks') mask visceral operational disruption. N-gram models and standard sentiment classifiers register positive polarity unless sarcasm markers or airport duration thresholds ('6 hours on the floor') are explicitly weighted."
        },
        {
            "failure_id": "FAIL_02",
            "category": "Compound Multi-Intent Collision",
            "customer_tweet": "My flight DL204 was cancelled in Detroit AND your baggage desk lost my suitcase containing my wedding dress for tomorrow!",
            "ground_truth_intent": "baggage_issue",
            "ground_truth_escalate": True,
            "root_cause_hypothesis": "Single-label classification forces an artificial winner between flight_disruption and baggage_issue. When multiple operational failures collide with high emotional urgency (wedding dress), resolving only one intent results in catastrophic customer dissatisfaction."
        },
        {
            "failure_id": "FAIL_03",
            "category": "Implicit Vulnerability Without Clinical Keywords",
            "customer_tweet": "My 84-year-old grandmother has been sitting alone by gate C12 for 4 hours with no one bringing the wheelchair we reserved.",
            "ground_truth_intent": "complaint_staff_service",
            "ground_truth_escalate": True,
            "root_cause_hypothesis": "Customer does not use acute medical alarm words like 'emergency', 'paramedic', or 'cardiac', but the vulnerability of an unattended octogenarian requires DOT Part 382 disability compliance escalation. Keyword-only filters completely miss this."
        },
        {
            "failure_id": "FAIL_04",
            "category": "Ambiguous 6-Letter Alphanumeric PII False Positives",
            "customer_tweet": "Flying ATLMSP on DL1420 next week. What terminal does it land in?",
            "ground_truth_intent": "inflight_airport_service",
            "ground_truth_escalate": False,
            "root_cause_hypothesis": "Airport pair codes ('ATLMSP') or promo codes can match the 6-character alphanumeric pattern of a Delta PNR record locator, triggering an unnecessary privacy escalation warning and demanding a DM when the query was harmless."
        },
        {
            "failure_id": "FAIL_05",
            "category": "Precedent Drift & Discretionary Compensation Hallucination",
            "customer_tweet": "Flight delayed 45 mins due to late incoming aircraft. Will I get a hotel room and food voucher?",
            "ground_truth_intent": "flight_disruption",
            "ground_truth_escalate": False,
            "root_cause_hypothesis": "Historical resolution databases contain rare goodwill vouchers issued by senior agents under extraordinary circumstances. A pure RAG retriever might retrieve an agent offering a $100 voucher and tempt the model to over-promise compensation for a routine 45-minute delay where DOT rules require none."
        }
    ]

    print("=" * 95)
    print("TOP 5 REAL-WORLD FAILURE MODES & ROOT CAUSE ANALYSIS")
    print("=" * 95)

    for item in failure_cases:
        clean_tweet = item["customer_tweet"].encode("ascii", "replace").decode()
        print(f"\n[{item['failure_id']}] Category: {item['category']}")
        print(f"Customer Tweet: \"{clean_tweet}\"")
        res = agent.process_tweet(item["customer_tweet"])
        print(f"Agent Prediction -> Intent: {res['intent']} (Conf: {res['intent_confidence']})")
        print(f"Agent Triage     -> Decision: {res['triage_decision']} | Reason: {res['escalation_reason'][:80]}...")
        print(f"Root Cause Hypothesis: {item['root_cause_hypothesis']}")
        print("-" * 95)

    out_path = os.path.join(base_dir, "data", "failure_modes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(failure_cases, f, indent=2)
    print(f"\nDetailed failure modes exported to {out_path}")

if __name__ == "__main__":
    analyze_top_failures()
