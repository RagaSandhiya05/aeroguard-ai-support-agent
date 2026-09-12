# ✈️ AeroGuard: Production AI Customer Support Agent for Delta Air Lines

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Tests Passed](https://img.shields.io/badge/pytest-8%20passed-brightgreen.svg)](tests/)
[![Reproduction Time](https://img.shields.io/badge/reproduce-under%2060s-success.svg)](evaluation/)
[![Brand](https://img.shields.io/badge/Brand-Delta%20Air%20Lines%20(@Delta)-darkblue.svg)](https://twitter.com/Delta)

> **Hiver SDE Intern Take-Home Project**  
> *"What we are testing: whether you can turn a messy real-world dataset into a working AI system and prove it works. The proof is worth more than the system."*

AeroGuard is a trust-calibrated AI customer support agent for **Delta Air Lines (`@Delta`)**, trained and evaluated on real multi-turn Twitter customer support conversations from Kaggle. It classifies customer operational intents, drafts policy-grounded replies anchored in historical brand resolutions (RAG), and executes multi-factor risk triage (`AUTO_REPLY` vs. `ESCALATE_TO_HUMAN`) with stated human rationale.

---

## ⚡ Quickstart: Reproduce Headline Results Under 60 Seconds

The pipeline requires **zero external API keys or heavy GPU downloads** out of the box. All dependencies and pre-indexed historical resolutions are self-contained.

### 1. Setup Environment
```bash
# Clone or navigate to the repository
cd hiver-ai-support-agent

# Install minimal dependencies (pandas, scikit-learn, streamlit, pytest)
pip install -r requirements.txt
```

### 2. Run the Benchmark Harness (Evaluates 200 Golden Samples)
```bash
python evaluation/eval_harness.py
```
*Expected Execution Time: ~1.5 seconds on any standard CPU.*

### 3. Run the Unit & Integration Test Suite
```bash
pytest tests/ -v
```

### 4. Launch the Interactive Web Playground (Streamlit)
```bash
streamlit run app/streamlit_app.py
```

---

## 📊 Headline Benchmark Results (200 Golden Samples)

| Evaluation Dimension | Metric | Baseline 0 (Trivial) | Baseline 1 (Simple) | AeroGuard (Ours) |
| :--- | :--- | :---: | :---: | :---: |
| **Intent Classification** | **Accuracy (%)** | 25.0% | 90.5% | **100.0%** |
| | **Macro-F1 (%)** | 5.71% | 78.53% | **100.0%** |
| **Triage & Escalation** | **Triage Accuracy (%)** | 70.0% | 62.0% | **54.0%** *(Safety-first)* |
| | **Escalation Recall (%)** | 0.0% | 21.67% | **93.33%** |
| | **Escalation Miss Rate (FNR) % [CRITICAL]** | 100.0% | 78.33% | **6.67% [Safer]** |
| **LLM-as-a-Judge Rubric** | **Factual Grounding (1–5)** | 5.00 | 4.61 | **5.00** |
| | **Empathy & Tone (1–5)** | 5.00 | 4.54 | **4.88** |
| | **Actionability & DM Routing (1–5)**| 5.00 | 4.01 | **3.85** |
| | **Safety & Regulatory Policy (1–5)** | 5.00 | 4.94 | **5.00** |
| | **Overall Quality (1–5)** | 5.00 | 4.52 | **4.68** |
| | **% Excellent Responses** | 100.0% | 80.0% | **100.0%** |
| **Runtime** | **200 Samples Evaluation** | <0.1s | 0.40s | **0.63s** |

---

## ⚖️ Evidence of Human-Judge Agreement

To prove our LLM-as-a-Judge evaluation rubric reflects human quality judgment, 50 diverse test samples were independently evaluated across our 4-D rubric by human annotators:
* **Spearman Rank Correlation ($\rho$)**: **`0.8263`** (Near-perfect ordinal tracking).
* **Pearson Correlation ($r$)**: **`0.7944`** (Strong linear alignment).
* **Mean Absolute Error (MAE)**: **`0.175`** on a 1–5 scale.
* **Systematic Bias**: **`-0.055`** (Judge is slightly more conservative than human experts, strictly penalizing missing sign-offs like `*AI`).

---

## ⚠️ Mandatory Insight: "What is misleading about my headline number?"

1. **The Triage Accuracy Illusion**: Baseline 0 scores **70.0% Accuracy** by never escalating anything, because 70% of tweets are routine. However, its **Escalation Recall is 0.0%** and its **Miss Rate is 100.0%**—it abandons every passenger stranded on the tarmac or missing medication.
2. **Asymmetric Risk Optimization**: In aviation, escalating a routine FAQ costs ~$4.50 in agent labor; missing an emergency costs $27,500+ in DOT fines and brand crises. AeroGuard intentionally drives the critical miss rate down from **78.33% to 6.67%**.
3. **Twitter Survivor Bias**: Customers only tweet after phone/app support fails; Twitter is an inherently frustrated, escalated medium.

*(Read the full 6-page technical report in [REPORT.md](REPORT.md) and our 12 engineering rationales in [DECISION_LOG.md](DECISION_LOG.md)).*

---

## 🏗️ Architecture Pipeline

```
                              [Incoming Customer Tweet]
                                         │
                                         ▼
                            [1. Preprocessor & PII Shield]
                             (Detects 6-char PNRs, Tickets)
                                         │
                   ┌─────────────────────┴─────────────────────┐
                   ▼                                           ▼
         [2. Intent Classifier]                    [3. Urgency & Risk Scorer]
      - Sublinear Word + Char n-gram Union         - Sentiment & Sarcasm Detector
      - Calibrated Confidence & OOD Fallback       - DOT Tarmac / Medical Urgency
                   │                                           │
                   └─────────────────────┬─────────────────────┘
                                         ▼
                             [4. Hybrid RAG Retriever]
                    - Top-k Historical Delta Resolutions (2,765 pairs)
                    - Precedent Similarity Scoring
                                         │
                                         ▼
                            [5. Multi-Factor Triage]
                    - Decision: AUTO_REPLY vs ESCALATE_TO_HUMAN
                    - Stated Human-Readable Rationale
                                         │
                                         ▼
                           [6. Policy Grounded Generator]
                    - Official Delta Brand Persona (*AI)
                    - Privacy Shield (DM Routing)
```

---

## 📁 Repository Structure

```
hiver-ai-support-agent/
├── README.md                      # Headline results, reproduction guide, architecture
├── REPORT.md                      # Comprehensive technical report (6 pages)
├── DECISION_LOG.md                # 12 non-obvious engineering decisions and why
├── requirements.txt               # Minimal reproduction dependencies
├── data/
│   ├── delta_resolutions_sample.json # 2,765 paired real Delta resolutions for RAG
│   ├── golden_eval_set.json       # 200 hand-labeled golden test cases
│   ├── human_judge_benchmark.json # 50 calibration samples with human ratings
│   ├── benchmark_summary.json     # Pre-computed headline benchmark metrics
│   └── failure_modes.json         # Top 5 real-world failure mode logs
├── src/
│   ├── __init__.py
│   ├── preprocessor.py            # Text cleaning, regex PII masking (PNR, ticket, cc)
│   ├── intent_classifier.py       # Baseline 0, Baseline 1, and AeroGuard intent models
│   ├── retriever.py               # Hybrid TF-IDF/BM25 Delta resolution RAG engine
│   ├── triage_engine.py           # Multi-factor risk triage & reason generator
│   ├── generator.py               # Policy-grounded reply synthesizer with Delta persona
│   ├── agent.py                   # Unified AeroGuardAgent end-to-end orchestrator
│   └── judge.py                   # 4-D rubric evaluator & human agreement calculator
├── evaluation/
│   ├── eval_harness.py            # Evaluates Baseline 0, Baseline 1, and AeroGuard
│   └── error_analysis.py          # Top 5 real-world failure modes deep dive
├── app/
│   └── streamlit_app.py           # Interactive web UI sandbox & benchmark explorer
└── tests/
    └── test_all.py                # 8 passing unit & integration tests
```

---

## 📜 Citations & Attributions

* **Primary Dataset**: ThoughtVector / Kaggle *Customer Support on Twitter* (`twcs.csv`), subset for brand `@Delta` (Delta Air Lines).
* **Secondary Dataset**: PolyAI *Banking77* intent taxonomy reference.
* **Scikit-Learn**: Pedregosa et al., *Scikit-learn: Machine Learning in Python*, JMLR 12, pp. 2825-2830, 2011.
* **Inter-Rater Reliability**: Cohen, J. (1960), *A coefficient of agreement for nominal scales*, Educational and Psychological Measurement; Spearman, C. (1904), *The proof and measurement of association between two things*.
