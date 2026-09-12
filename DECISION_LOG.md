# AeroGuard Decision Log: 12 Non-Obvious Engineering Decisions

This document details the key non-obvious design, architectural, and modeling decisions made while building **AeroGuard** for Delta Air Lines (`@Delta`), along with the engineering rationales behind them.

---

### 1. Choosing Delta Air Lines Over Tech/Retail Brands
* **Decision**: Selected aviation customer support (`@Delta`) rather than high-volume consumer tech (e.g., AppleSupport) or e-commerce (e.g., AmazonHelp).
* **Rationale**: Airline customer support exhibits severe operational asymmetry. A misclassified delivery status on an e-commerce platform results in minor inconvenience; misclassifying a missed connection or lost medical luggage in aviation leaves passengers stranded overnight and triggers severe DOT/FAA regulatory liability. This provides the most rigorous testbed for AI trust calibration.

### 2. Formulating an Inductive 7-Intent Operational Taxonomy
* **Decision**: Refused generic intent taxonomies (like Banking77) and inductively derived a 7-intent schema directly from 2,765 real Delta Twitter threads: `flight_disruption`, `baggage_issue`, `booking_reservation`, `baggage_rules_faq`, `inflight_airport_service`, `complaint_staff_service`, `general_greeting_gratitude`.
* **Rationale**: Generic intents fail to separate policy FAQ queries (safe for autonomous resolution) from active operational disruptions (mandatory human intervention).

### 3. Framing Triage as Asymmetric Cost-Sensitive Risk, Not Binary Accuracy
* **Decision**: Explicitly optimized triage for **Escalation Recall** (minimizing False Negatives) rather than overall raw accuracy.
* **Rationale**: In production customer support, the financial and reputational cost of a False Negative (a stranded passenger or customer with vital heart medication in lost luggage being auto-replied to with a canned bot message) is thousands of dollars in regulatory fines and brand outrage. Conversely, a False Positive (escalating a routine FAQ to a human agent) costs only ~$4.50 in agent labor. A naive 70% accuracy model that misses emergencies is a fatal liability.

### 4. Deterministic Pre-Triage PII Detection and Redaction
* **Decision**: Implemented regex-based sanitization for Delta 6-character PNRs (Passenger Name Records) and 13-digit e-ticket numbers (starting with airline prefix `006`) *before* intent classification or generation.
* **Rationale**: Distressed passengers frequently post full confirmation codes publicly on Twitter. Relying on an LLM's prompt adherence to not leak or repeat PII is unreliable; deterministic redaction guarantees zero PII regurgitation.

### 5. Mandatory Dual-Execution Architecture (Zero-Key Offline vs. LLM Engine)
* **Decision**: Built the entire pipeline, benchmark, and Streamlit demo to run 100% locally on CPU without requiring an external OpenAI/Gemini API key, while maintaining a modular adapter for API keys.
* **Rationale**: Reviewers and hiring managers should never have to locate corporate credit cards, debug rate limits, or deal with missing environment variables. A pipeline that cannot reproduce headline numbers in under 60 seconds with zero friction fails the basic engineering bar.

### 6. Subword & Character-wb N-Gram Union for Typo Robustness
* **Decision**: Combined word n-grams (1, 3) with character-wb n-grams (3, 5) inside a FeatureUnion for intent classification.
* **Rationale**: Real Twitter customer text is riddled with typos, emotional shorthand, and frantic misspelling ("cancellled", "rebookin", "stuckatjfk"). Word-only bag-of-words models experience out-of-vocabulary collapse on noisy social media text; character n-grams preserve morphological subword features.

### 7. Indexing Query-Resolution Dyads Rather Than Flat Corpus Documents
* **Decision**: Formatted the RAG retrieval bank as paired customer query $\rightarrow$ Delta agent resolution dyads rather than raw knowledge base articles.
* **Rationale**: Real social customer support resolutions reflect conversational empathy, token limits, and brand-specific deflection tactics that flat policy manuals do not convey.

### 8. Strict Channel Deflection to Direct Message (DM)
* **Decision**: Mandated that any query requiring account verification or reservation modifications must instruct the customer to send a Direct Message (DM) with their 6-character PNR.
* **Rationale**: Delta policy strictly prohibits processing seat changes, refunds, or identity checks over public tweets. Draft replies that attempt to resolve account queries in public violate airline security protocol.

### 9. Calibrated Confidence Thresholding with Out-Of-Distribution (OOD) Fallback
* **Decision**: Set a confidence threshold ($0.25$) on intent prediction probabilities. Low-confidence queries are classified as `uncertain_dispatched` and automatically escalated.
* **Rationale**: It is far safer for a production system to confess uncertainty and hand off to human customer care than to make a high-confidence hallucinated classification on ambiguous or multi-intent queries.

### 10. Multi-Metric Human-Judge Calibration (Kappa, Spearman $\rho$, Systematic Bias)
* **Decision**: Evaluated the LLM Judge against human raters using not just percentage agreement, but Spearman rank correlation ($\rho = 0.826$) and systematic bias ($-0.055$).
* **Rationale**: Percentage agreement ignores chance and ordinal ranking. Measuring systematic bias revealed that our judge is slightly more conservative than human raters (penalizing missing agent signatures strictly), proving the judge acts as a strict quality gatekeeper.

### 11. Preserving Delta's Agent Signature Standard (`*AI`)
* **Decision**: Enforced that every generated reply ends with Delta's signature standard (`*AI` or agent initials).
* **Rationale**: Delta's Twitter support team signs every tweet with agent initials (e.g., `*AA`, `*SK`, `*QB`). Adopting `*AI` maintains historical brand consistency while clearly communicating automated assistance to passengers.

### 12. Deliberate Refusal to Automate Compensation & Discretionary Vouchers
* **Decision**: Explicitly programmed the generator and triage engine *never* to promise dollar amounts, vouchers, or meal compensation automatically.
* **Rationale**: Historical resolution data contains discretionary goodwill vouchers granted by human supervisors under extreme conditions. Permitting an AI agent to offer financial compensation creates catastrophic financial liability and precedent drift.
