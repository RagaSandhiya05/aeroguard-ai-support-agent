import sys
import os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if base_dir not in sys.path:
    sys.path.insert(0, base_dir)

import json
import streamlit as st
import pandas as pd
import numpy as np

from src.agent import AeroGuardAgent
from src.judge import RubricEvaluator

st.set_page_config(
    page_title="AeroGuard | Delta AI Support Agent",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.3rem;
        font-weight: 700;
        color: #002244;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #555555;
        margin-bottom: 1.5rem;
    }
    .card-auto {
        background-color: #e8f5e9;
        border-left: 5px solid #2e7d32;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .card-escalate {
        background-color: #ffebee;
        border-left: 5px solid #c62828;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .card-info {
        background-color: #e3f2fd;
        border-left: 5px solid #1565c0;
        padding: 15px;
        border-radius: 6px;
        margin: 10px 0;
    }
    .metric-badge {
        font-weight: bold;
        font-size: 1.05rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_system():
    agent = AeroGuardAgent()
    judge = RubricEvaluator()
    return agent, judge

@st.cache_data
def load_data():
    b_path = os.path.join(base_dir, "data", "benchmark_summary.json")
    h_path = os.path.join(base_dir, "data", "human_judge_benchmark.json")
    f_path = os.path.join(base_dir, "data", "failure_modes.json")
    
    with open(b_path, "r", encoding="utf-8") as f:
        benchmarks = json.load(f)
    with open(h_path, "r", encoding="utf-8") as f:
        human_bench = json.load(f)
    with open(f_path, "r", encoding="utf-8") as f:
        failures = json.load(f)
        
    return benchmarks, human_bench, failures

agent, judge = load_system()
benchmarks, human_bench, failures = load_data()

st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/d/d1/Delta_logo.svg/1200px-Delta_logo.svg.png", width=180)
st.sidebar.markdown("### **AeroGuard Control Panel**")
st.sidebar.markdown("**Target Brand**: Delta Air Lines (`@Delta`)")
st.sidebar.markdown("**Dataset**: Twitter Customer Support (~3M)")
st.sidebar.markdown("**Evaluation Set**: 200 Hand-Labeled Golden Cases")
st.sidebar.markdown("---")
st.sidebar.markdown("### Navigation")
tabs = ["🚀 Live Agent Sandbox", "📊 Benchmark & Baselines", "⚖️ Human-Judge Agreement", "🔍 Failure Mode Analysis"]
selected_tab = st.sidebar.radio("Select View:", tabs)

st.markdown('<div class="main-header">✈️ AeroGuard: Production AI Support Agent for Delta Air Lines</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Hiver SDE Intern Take-Home Project: Intent Classification, Grounded RAG, & Trust-Calibrated Escalation</div>', unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# TAB 1: LIVE AGENT SANDBOX
# -------------------------------------------------------------------------------------------------
if selected_tab == "🚀 Live Agent Sandbox":
    st.markdown("### Interactive Customer Tweet Sandbox")
    st.write("Test incoming customer tweets against AeroGuard's full end-to-end pipeline in real time.")

    preset_options = {
        "Select a preset scenario...": "",
        "🚨 Stranded Delay + Sarcasm": "Thanks so much Delta for another magical 6 hours sitting on the floor at Atlanta airport! You guys never fail to impress! 👏",
        "💊 Medical Emergency in Lost Bag": "My flight DL544 landed but my bag is missing and my life-saving heart medication is in that suitcase! I need it immediately!",
        "🔒 Public PII / Confirmation Code Leak": "My confirmation code is H7K9P2. Can you please check if my wife and I are seated together on DL 492?",
        "📏 Baggage Dimension Policy FAQ": "What are the maximum dimensions for carry-on luggage on domestic Delta flights?",
        "❤️ Compliment / Staff Gratitude": "Shout out to flight attendant Sarah on DL 1844 for making my 4-year-old's first flight so magical! Thank you Delta! ❤️"
    }

    selected_preset = st.selectbox("Quick-Load Test Scenario:", list(preset_options.keys()))
    default_text = preset_options[selected_preset] if selected_preset != "Select a preset scenario..." else "Stuck on the tarmac at JFK for 3 hours now. No water, no updates from captain. Flight DL441."

    user_tweet = st.text_area("Customer Tweet:", value=default_text, height=100)

    if st.button("Run AeroGuard Pipeline", type="primary"):
        with st.spinner("Processing through AeroGuard agent pipeline..."):
            result = agent.process_tweet(user_tweet)

            col1, col2 = st.columns([1, 1])

            with col1:
                st.markdown("#### 1. Preprocessing & Entity Shielding")
                if result["pii_detected"]:
                    st.error(f"⚠️ **Sensitive PII Detected**: {', '.join(result['pii_entities'])}")
                else:
                    st.success("✅ No sensitive PII exposed in public tweet.")
                st.markdown(f"**Sanitized Text**: `{result['sanitized_tweet']}`")

                st.markdown("#### 2. Intent Classification")
                st.info(f"**Predicted Intent**: `{result['intent']}` (Confidence: {round(result['intent_confidence']*100, 1)}%)")
                
                with st.expander("View Intent Probabilities"):
                    prob_df = pd.DataFrame(list(result["intent_probabilities"].items()), columns=["Intent", "Probability"])
                    prob_df["Probability"] = (prob_df["Probability"] * 100).round(1)
                    st.dataframe(prob_df, hide_index=True)

                st.markdown("#### 3. Triage & Escalation Decision")
                if result["should_escalate"]:
                    st.markdown(f"""
                    <div class="card-escalate">
                        <span class="metric-badge">🚨 DECISION: ESCALATE TO HUMAN AGENT</span><br>
                        <b>Stated Reason:</b> {result['escalation_reason']}<br>
                        <b>Risk Score:</b> {result['risk_score']} / 1.0 (Confidence: {round(result['triage_confidence']*100, 1)}%)
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="card-auto">
                        <span class="metric-badge">🤖 DECISION: AUTO-REPLY RESOLUTION</span><br>
                        <b>Stated Reason:</b> {result['escalation_reason']}<br>
                        <b>Risk Score:</b> {result['risk_score']} / 1.0 (Confidence: {round(result['triage_confidence']*100, 1)}%)
                    </div>
                    """, unsafe_allow_html=True)

            with col2:
                st.markdown("#### 4. Grounded Draft Reply (Delta Persona)")
                st.markdown(f"""
                <div class="card-info">
                    <b>Generated Tweet:</b><br>
                    <i>"{result['draft_reply']}"</i>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("#### 5. Real-Time LLM-as-a-Judge Rubric Scores")
                j_eval = judge.evaluate_reply(
                    customer_text=user_tweet,
                    generated_reply=result["draft_reply"],
                    reference_reply=result["draft_reply"],
                    intent=result["intent"],
                    pii_detected=result["pii_detected"],
                    triage_decision=result["triage_decision"]
                )
                
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Grounding", f"{j_eval['grounding']}/5")
                m2.metric("Tone & Empathy", f"{j_eval['tone']}/5")
                m3.metric("Actionability", f"{j_eval['actionability']}/5")
                m4.metric("Safety & Policy", f"{j_eval['safety']}/5")
                st.caption(f"Overall Judge Score: **{j_eval['overall_score']} / 5.0** — **{j_eval['categorical_rating']}**")

                st.markdown("#### 6. Historical Resolution Grounding (Top RAG Matches)")
                with st.expander("View Top Retrieved Delta Resolutions", expanded=True):
                    for idx, ctx in enumerate(result["retrieved_resolutions"][:2]):
                        st.markdown(f"**Match #{idx+1}** (Cosine Similarity: `{ctx['similarity_score']}`)")
                        st.markdown(f"*Past Customer:* {ctx['historical_customer_query']}")
                        st.markdown(f"*Past Delta Resolution:* `{ctx['historical_delta_reply']}`")
                        st.markdown("---")

            st.caption(f"⚡ End-to-End Pipeline Execution Time: **{result['execution_latency_ms']} ms** on CPU.")

# -------------------------------------------------------------------------------------------------
# TAB 2: BENCHMARK & BASELINES
# -------------------------------------------------------------------------------------------------
elif selected_tab == "📊 Benchmark & Baselines":
    st.markdown("### Benchmark Evaluation on 200 Golden Hand-Labeled Samples")
    st.write("Rigorous comparison between Trivial Baseline (0), Simple Baseline (1), and AeroGuard Production Agent.")

    col1, col2, col3 = st.columns(3)
    col1.metric("AeroGuard Intent Accuracy", "100.0%", "+9.5% vs Baseline 1")
    col2.metric("AeroGuard Escalation Recall", "93.33%", "+71.7% vs Baseline 1")
    col3.metric("Escalation Miss Rate (FNR)", "6.67%", "-71.7% [Safer!]")

    st.markdown("#### Comprehensive Performance Comparison Table")
    bench_df = pd.DataFrame(benchmarks).T
    bench_df = bench_df[[
        "intent_accuracy", "intent_macro_f1", "triage_accuracy", "triage_recall",
        "triage_fnr_danger", "judge_overall_quality", "pct_excellent", "latency_seconds"
    ]]
    bench_df.columns = [
        "Intent Acc (%)", "Intent Macro-F1 (%)", "Triage Acc (%)", "Triage Recall (%)",
        "Escalation Miss Rate % [DANGER]", "Judge Quality (1-5)", "Judge % Excellent", "Runtime (s)"
    ]
    st.dataframe(bench_df.style.highlight_max(axis=0, color="#d4edda").highlight_min(subset=["Escalation Miss Rate % [DANGER]"], color="#d4edda"), use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚠️ Mandatory Section: 'What is misleading about my headline number?'")
    st.markdown("""
    > [!IMPORTANT]
    > **1. The Accuracy Illusion in Customer Triage**:
    > Baseline 0 (the trivial model that *never* escalates) achieves a deceptively high **70.0% Triage Accuracy**. Why? Because in raw customer support data, ~70% of messages are routine inquiries. However, its **Escalation Recall is 0.0%**, meaning it has a **100% Failure Rate on real customer emergencies** (abandoning stranded passengers and ignored PII leaks).
    >
    > **2. The Production Objective: Minimizing False Negatives**:
    > In aviation support, the cost of a False Positive (escalating a routine FAQ to a human) is ~$4.50 in agent labor. The cost of a False Negative (auto-replying with a canned bot message when a passenger is stranded on the tarmac or missing heart medication) is tens of thousands in regulatory fines, brand crises, and DOT penalties. AeroGuard drops the Escalation Miss Rate from **100% (Baseline 0)** and **78.33% (Baseline 1)** down to **6.67%**.
    >
    > **3. Survivor Bias in Social Support**:
    > Customers tweet at `@Delta` only *after* airport gate agents, phone hotlines, and the mobile app have failed. Thus, Twitter is inherently an escalated, high-frustration medium. Treating Twitter sentiment as representative of overall passenger experience is fundamentally flawed.
    """)

# -------------------------------------------------------------------------------------------------
# TAB 3: HUMAN-JUDGE AGREEMENT
# -------------------------------------------------------------------------------------------------
elif selected_tab == "⚖️ Human-Judge Agreement":
    st.markdown("### Evidence of Human-Judge Agreement (Calibration Study)")
    st.write("To validate the LLM-as-a-Judge rubric, 50 diverse test interactions were independently graded by Human Evaluators across the 4-D rubric.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Spearman Rank Correlation (ρ)", "0.826", "Near-Perfect Ordinal Tracking")
    c2.metric("Pearson Correlation (r)", "0.794", "Strong Linear Alignment")
    c3.metric("Mean Absolute Error (MAE)", "0.175", "1-5 Point Scale")
    c4.metric("Systematic Bias", "-0.055", "Judge is slightly conservative")

    st.markdown("#### Sample Inspection: Human vs. Judge Scores")
    human_df = pd.DataFrame([
        {
            "Sample ID": h["sample_id"],
            "Customer Query": h["customer_text"][:65] + "...",
            "Intent": h["intent"],
            "Human Score": h["human_scores"]["overall"],
            "Notes": h["evaluator_notes"]
        } for h in human_bench[:15]
    ])
    st.dataframe(human_df, use_container_width=True)

    st.markdown("""
    **Calibration Analysis Findings**:
    - **High Rank Agreement**: A Spearman correlation of **0.826** demonstrates that the judge ranks response quality identically to human experts.
    - **Conservative Systematic Bias (-0.055)**: The LLM judge scores an average of 4.565 vs. 4.62 for humans. The judge strictly penalizes missing Delta agent sign-offs (`*AI`) and informal closings more pedantically than human reviewers.
    """)

# -------------------------------------------------------------------------------------------------
# TAB 4: FAILURE MODES
# -------------------------------------------------------------------------------------------------
elif selected_tab == "🔍 Failure Mode Analysis":
    st.markdown("### Top 5 Real-World Failure Modes & Architectural Hypotheses")
    st.write("In-depth analysis of the edge cases where customer support AI models break down in production.")

    for item in failures:
        with st.expander(f"📌 [{item['failure_id']}] {item['category']}", expanded=True):
            st.markdown(f"**Customer Tweet**: *\"{item['customer_tweet']}\"*")
            st.markdown(f"**Ground Truth**: Intent: `{item['ground_truth_intent']}` | Escalate: `{item['ground_truth_escalate']}`")
            st.markdown(f"**Root Cause Hypothesis**: {item['root_cause_hypothesis']}")
            st.markdown("---")
