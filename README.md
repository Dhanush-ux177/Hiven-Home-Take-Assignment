# OmniSupport AI Agent & Benchmark Suite

An enterprise-grade, grounded customer support AI agent and evaluation benchmark suite built on authentic customer support Twitter dialogues (Kaggle: `thoughtvector/customer-support-on-twitter` - `@AppleSupport` corpus).

The system features multi-intent classification, safety and PII escalation guardrails, BM25 historical grounding retrieval, a 200-example golden benchmark, comparative baselines, LLM-as-a-judge calibration, and a responsive mobile-ready web interface.

---

## 100% Pure Python Architecture

The core engine, evaluation harness, baselines, diagnostic CLI, unit tests, and HTTP API server are implemented in **100% standard library Python 3** with **zero external pip dependencies required** (`http.server`, `urllib`, `json`, `math`, `re`, `argparse`, `unittest`).

### Command-Line Interface (`main.py`)

```bash
# 1. Run the full benchmark (200 golden examples vs 2 baselines) in < 10 seconds:
python3 main.py

# 2. Launch interactive diagnostic terminal REPL:
python3 main.py --interactive

# 3. Test a custom incoming customer tweet with comparative baselines:
python3 main.py --query "My Apple ID is locked and I can't receive my 2FA code!"

# 4. Output machine-readable JSON for single queries:
python3 main.py --query "Keyboard lag on iOS 11" --json

# 5. Run the automated Python unit test suite:
python3 main.py --test

# 6. Display Kaggle dataset taxonomy and escalation statistics:
python3 main.py --stats

# 7. Run a quick evaluation on 50 sample cases:
python3 main.py --quick

# 8. Start the pure Python HTTP server on port 3000:
python3 main.py --serve --port 3000
# or directly:
python3 server.py
```

---

## Headline Benchmark Results

Evaluated across **200 hand-labelled real-world customer conversations** from Kaggle:

| Metric | Baseline 1: Trivial | Baseline 2: Simple | Proposed Agent | Operational Target |
| :--- | :---: | :---: | :---: | :---: |
| **Intent Accuracy** | 50.0% | 75.5% | **78.0%** | > 75.0% |
| **Intent Macro F1** | 0.133 | 0.563 | **0.733** | > 0.700 |
| **Escalation Accuracy** | 75.0% | 71.0% | **73.0%** | > 70.0% |
| **Escalation F1** | 0.000 | 0.147 | **0.565** | > 0.500 |
| **False Auto-Handle Rate (Safety)** | **100.0%** *(Fatal)* | **90.0%** *(High)* | **30.0%** *(Controlled)* | **< 35.0%** |
| **Unnecessary Escalation Rate (Cost)** | 0.0% | 8.7% | 26.0% | < 30.0% |
| **Judge Quality Score (1-5 Scale)** | 4.12 | 3.66 | **3.99** | > 3.80 |
| **Judge-Human Adjacent Agreement** | 87.0% | 80.5% | **94.0%** | > 90.0% |
| **Deterministic Latency** | < 1 ms | < 1 ms | **< 2 ms** | < 100 ms |

---

## System Architecture

```
                 Incoming Customer Tweet
                            │
        ┌───────────────────┴───────────────────┐
        ▼                                       ▼
┌───────────────────────┐               ┌───────────────────────┐
│   Intent Classifier   │               │   Escalation Engine   │
│ (6 Empirical Intents) │               │(Guardrails & Policies)│
└───────────┬───────────┘               └───────────┬───────────┘
            │                                       │
            │ Intent Label + Confidence             │ Escalate: Bool + Reason
            ▼                                       ▼
┌───────────────────────────────────────────────────────────────┐
│              Historical Retriever (BM25 RAG)                  │
│       Retrieves top authentic historical support resolutions  │
└───────────────────────────────┬───────────────────────────────┘
                                │
                                ▼
┌───────────────────────────────────────────────────────────────┐
│                     Brand Reply Drafter                       │
│    Calm, empathetic, diagnostic isolation + Verified KB links │
└───────────────────────────────────────────────────────────────┘
```

1. **Intent Taxonomy:**
   - `SOFTWARE_OS_BUG`: iOS glitches, keyboard lag, update anomalies
   - `HARDWARE_BATTERY`: Battery health, rapid drain, physical defects, swollen cells
   - `ACCOUNT_SECURITY_ICLOUD`: Locked Apple IDs, 2FA codes, billing charges, phishing
   - `DEVICE_SETUP_FEATURE`: 3D touch, deleting apps, AirPods pairing, Family Sharing
   - `STORE_REPAIR_ORDER`: Genius Bar appointments, repair status, trade-in logistics
   - `GENERAL_FEEDBACK_RANT`: Customer venting, product design feedback

2. **Domain-Safe Escalation Policy:**
   - **Critical Guardrails:** PII protection (zero password or credential requests on public social media) and physical hardware safety (swollen battery / thermal hazards) trigger immediate escalation to private DM or physical store inspection.
   - **Self-Service Routing:** Standard single-turn software isolation questions and official knowledge base documentation are auto-handled without burdening human queues.

3. **Grounding & Historical Retrieval:**
   - Retrieves top matching resolutions from a curated bank of 51 authentic support interactions (`data/historical_exemplars.json`).

---

## Project Structure

```
├── main.py                     # Unified CLI entrypoint (eval, query, REPL, stats, tests)
├── server.py                   # Pure Python HTTP server (threading, REST API, static files)
├── REPORT.md                   # Full 6-page Technical Report & Findings
├── README.md                   # System documentation & quickstart
├── requirements.txt            # Zero external pip requirements documented
├── package.json                # Frontend build scripts & dev server runner
├── data/
│   ├── golden_eval_set.json    # 200 hand-labelled Kaggle test conversations
│   ├── historical_exemplars.json# 51 authentic support grounding cases
│   ├── sampling_note.md        # Dataset stratification & labelling methodology
│   └── evaluation_results.json # Output benchmark metrics & failure modes
├── scripts/
│   ├── agent.py                # Core OmniSupport Agent implementation
│   ├── baselines.py            # Baseline 1 (Trivial) & Baseline 2 (Simple)
│   ├── evaluator.py            # Evaluation harness, metrics, Cohen's Kappa
│   ├── run_eval.py             # Evaluation runner helper
│   └── prepare_dataset.py      # Raw Kaggle dataset extraction & cleaning
├── tests/
│   └── test_agent.py           # Python unit tests for agent, guardrails, & baselines
├── src/
│   ├── App.tsx                 # Mobile-responsive web dashboard & evaluation explorer
│   ├── main.tsx                # React entry point
│   └── index.css               # Clean Tailwind styling
└── dist/                       # Production-compiled web dashboard assets
```

---

## Detailed Technical Report

For the in-depth discussion covering problem framing, deliberate non-goals, the "Twitter DM Deflection" ground truth artifact, and top 5 failure mode hypotheses, read [REPORT.md](REPORT.md).
