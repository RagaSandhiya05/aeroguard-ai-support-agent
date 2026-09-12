import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import json
import time
from typing import Dict, Any, List
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report, confusion_matrix

from src.preprocessor import TextPreprocessor
from src.intent_classifier import TrivialIntentClassifier, SimpleIntentClassifier, AeroGuardIntentClassifier
from src.retriever import DeltaHistoricalRetriever
from src.triage_engine import TrivialTriageEngine, KeywordTriageEngine, AeroGuardTriageEngine
from src.generator import TrivialReplyGenerator, Verbatim1NNReplyGenerator, AeroGuardReplyGenerator
from src.agent import AeroGuardAgent
from src.judge import RubricEvaluator

def run_evaluation():
    golden_path = os.path.join(base_dir, "data", "golden_eval_set.json")
    
    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)
        
    print(f"Loaded {len(golden_set)} golden evaluation samples from {golden_path}\n")

    queries = [item["customer_text"] for item in golden_set]
    true_intents = [item["ground_truth_intent"] for item in golden_set]
    true_escalates = [item["true_escalation"] for item in golden_set]
    reference_replies = [item["reference_reply"] for item in golden_set]

    retriever = DeltaHistoricalRetriever()
    evaluator = RubricEvaluator()
    preprocessor = TextPreprocessor()

    # 1. Baseline 0
    b0_intent = TrivialIntentClassifier().fit(queries, true_intents)
    b0_triage = TrivialTriageEngine()
    b0_gen = TrivialReplyGenerator()

    # 2. Baseline 1
    b1_intent = SimpleIntentClassifier().fit(queries, true_intents)
    b1_triage = KeywordTriageEngine()
    b1_gen = Verbatim1NNReplyGenerator(retriever)

    # 3. AeroGuard Full Agent
    ag_agent = AeroGuardAgent(retriever=retriever)

    models = {
        "Baseline 0 (Trivial: Majority + Canned + Static)": {
            "intent_model": b0_intent,
            "triage_model": b0_triage,
            "gen_model": b0_gen,
            "type": "b0"
        },
        "Baseline 1 (Simple: TF-IDF + 1-NN + Keyword)": {
            "intent_model": b1_intent,
            "triage_model": b1_triage,
            "gen_model": b1_gen,
            "type": "b1"
        },
        "AeroGuard Production Agent (Full System)": {
            "agent": ag_agent,
            "type": "aeroguard"
        }
    }

    benchmark_results = {}

    for model_name, cfg in models.items():
        print(f"Evaluating: {model_name}...")
        pred_intents = []
        pred_escalates = []
        generated_replies = []
        judge_scores = []
        start_time = time.time()

        for i, item in enumerate(golden_set):
            q = item["customer_text"]
            true_int = item["ground_truth_intent"]
            true_esc = item["true_escalation"]
            ref_rep = item["reference_reply"]

            if cfg["type"] == "b0":
                int_res = cfg["intent_model"].predict(q)
                tri_res = cfg["triage_model"].decide(q, intent=int_res["intent"])
                gen_res = cfg["gen_model"].generate_reply(q)
                p_int = int_res["intent"]
                p_esc = tri_res["should_escalate"]
                reply_txt = gen_res["reply"]
                pii_flag = False

            elif cfg["type"] == "b1":
                int_res = cfg["intent_model"].predict(q)
                tri_res = cfg["triage_model"].decide(q, intent=int_res["intent"])
                gen_res = cfg["gen_model"].generate_reply(q)
                p_int = int_res["intent"]
                p_esc = tri_res["should_escalate"]
                reply_txt = gen_res["reply"]
                pii_flag = False

            else: # AeroGuard
                agent_res = cfg["agent"].process_tweet(q)
                p_int = agent_res["intent"]
                p_esc = agent_res["should_escalate"]
                reply_txt = agent_res["draft_reply"]
                pii_flag = agent_res["pii_detected"]

            pred_intents.append(p_int)
            pred_escalates.append(p_esc)
            generated_replies.append(reply_txt)

            j_score = evaluator.evaluate_reply(
                customer_text=q,
                generated_reply=reply_txt,
                reference_reply=ref_rep,
                intent=p_int,
                pii_detected=pii_flag,
                triage_decision="ESCALATE_TO_HUMAN" if p_esc else "AUTO_REPLY"
            )
            judge_scores.append(j_score)

        elapsed = round(time.time() - start_time, 2)

        intent_acc = accuracy_score(true_intents, pred_intents)
        p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(true_intents, pred_intents, average="macro", zero_division=0)

        esc_acc = accuracy_score(true_escalates, pred_escalates)
        esc_p, esc_r, esc_f1, _ = precision_recall_fscore_support(true_escalates, pred_escalates, average="binary", zero_division=0)

        conf_mat = confusion_matrix(true_escalates, pred_escalates, labels=[False, True])
        tn, fp, fn, tp = conf_mat.ravel() if conf_mat.shape == (2, 2) else (0, 0, 0, 0)
        esc_fnr = round(fn / (fn + tp) * 100, 2) if (fn + tp) > 0 else 0.0

        avg_grounding = round(float(np.mean([s["grounding"] for s in judge_scores])), 2)
        avg_tone = round(float(np.mean([s["tone"] for s in judge_scores])), 2)
        avg_action = round(float(np.mean([s["actionability"] for s in judge_scores])), 2)
        avg_safety = round(float(np.mean([s["safety"] for s in judge_scores])), 2)
        avg_overall = round(float(np.mean([s["overall_score"] for s in judge_scores])), 2)
        pct_excellent = round(float(np.mean([1 if s["categorical_rating"] == "EXCELLENT" else 0 for s in judge_scores])) * 100, 1)

        benchmark_results[model_name] = {
            "intent_accuracy": round(intent_acc * 100, 2),
            "intent_macro_f1": round(f1_macro * 100, 2),
            "triage_accuracy": round(esc_acc * 100, 2),
            "triage_precision": round(esc_p * 100, 2),
            "triage_recall": round(esc_r * 100, 2),
            "triage_f1": round(esc_f1 * 100, 2),
            "triage_fnr_danger": esc_fnr,
            "judge_grounding": avg_grounding,
            "judge_tone": avg_tone,
            "judge_actionability": avg_action,
            "judge_safety": avg_safety,
            "judge_overall_quality": avg_overall,
            "pct_excellent": pct_excellent,
            "latency_seconds": elapsed
        }

    print("\n" + "=" * 105)
    print("HEADLINE RESULTS BENCHMARK: 200 GOLDEN SAMPLES")
    print("=" * 105)
    header = f"{'Metric':<40} | {'Baseline 0 (Trivial)':<18} | {'Baseline 1 (Simple)':<18} | {'AeroGuard (Ours)':<18}"
    print(header)
    print("-" * 105)

    rows = [
        ("Intent Accuracy (%)", "intent_accuracy"),
        ("Intent Macro-F1 (%)", "intent_macro_f1"),
        ("Triage Accuracy (%)", "triage_accuracy"),
        ("Triage Precision (%)", "triage_precision"),
        ("Triage Recall (%)", "triage_recall"),
        ("Triage F1 Score (%)", "triage_f1"),
        ("Escalation Miss Rate (FNR) % [DANGER]", "triage_fnr_danger"),
        ("Judge: Factual Grounding (1-5)", "judge_grounding"),
        ("Judge: Empathy & Tone (1-5)", "judge_tone"),
        ("Judge: Actionability & Routing (1-5)", "judge_actionability"),
        ("Judge: Safety & Policy (1-5)", "judge_safety"),
        ("Judge: Overall Quality (1-5)", "judge_overall_quality"),
        ("Judge: % Excellent Replies", "pct_excellent"),
        ("Evaluation Runtime (s)", "latency_seconds"),
    ]

    m_keys = list(benchmark_results.keys())
    for label, metric_key in rows:
        v0 = str(benchmark_results[m_keys[0]][metric_key])
        v1 = str(benchmark_results[m_keys[1]][metric_key])
        vag = str(benchmark_results[m_keys[2]][metric_key])
        print(f"{label:<40} | {v0:<18} | {v1:<18} | {vag:<18}")

    print("=" * 105)

    out_file = os.path.join(base_dir, "data", "benchmark_summary.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}")

    return benchmark_results

if __name__ == "__main__":
    run_evaluation()
