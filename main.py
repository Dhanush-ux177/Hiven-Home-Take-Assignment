#!/usr/bin/env python3
"""
main.py - @AppleSupport Grounded AI Customer Support Agent
Single Python Entrypoint for Inference, Evaluation, Interactive REPL, Testing, and Web Server.

Usage:
  python3 main.py                       # Run headline benchmark (200 golden cases vs 2 baselines)
  python3 main.py --query "My Apple ID is locked" # Run live inference
  python3 main.py --interactive         # Launch interactive diagnostic terminal
  python3 main.py --quick               # Fast evaluation on 50 sample cases
  python3 main.py --serve --port 3000   # Start pure Python web server
  python3 main.py --test                # Run test suite
  python3 main.py --stats               # View Kaggle dataset breakdown
"""

import os
import sys
import json
import argparse
import unittest

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from agent import AppleSupportAgent
from baselines import BaselineTrivial, BaselineSimple
from evaluator import run_full_benchmark, BenchmarkEvaluator

def print_banner():
    banner = """
========================================================================
     @AppleSupport Grounded AI Customer Support Agent (Python)
    Kaggle Benchmark: thoughtvector/customer-support-on-twitter
========================================================================
"""
    print(banner)

def show_dataset_stats():
    data_path = os.path.join(BASE_DIR, "data", "golden_eval_set.json")
    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    intents = {}
    escalates = {"True": 0, "False": 0}
    depths = []
    scores = []

    for item in data:
        intent = item.get("gold_intent", "UNKNOWN")
        intents[intent] = intents.get(intent, 0) + 1
        esc = str(item.get("gold_escalate", False))
        escalates[esc] = escalates.get(esc, 0) + 1
        depths.append(item.get("thread_depth", 1))
        scores.append(item.get("human_quality_score", 4.0))

    print(f"\n--- Kaggle Hand-Labelled Golden Dataset Statistics (N = {len(data)}) ---")
    print(f"\n[Intent Distribution]:")
    for intent, count in sorted(intents.items(), key=lambda x: -x[1]):
        pct = (count / len(data)) * 100
        bar = "█" * int(pct / 3)
        print(f"  {intent:26} : {count:3d} ({pct:5.1f}%) | {bar}")

    print(f"\n[Escalation Distribution]:")
    print(f"  Escalate to Human (True)  : {escalates['True']:3d} ({(escalates['True']/len(data))*100:.1f}%)")
    print(f"  Auto-Handle (False)       : {escalates['False']:3d} ({(escalates['False']/len(data))*100:.1f}%)")

    print(f"\n[Conversation Depth]:")
    print(f"  Single-turn inquiries     : {sum(1 for d in depths if d == 1)} cases")
    print(f"  Multi-turn threads (2+)   : {sum(1 for d in depths if d > 1)} cases")
    print(f"  Mean Human Quality Score  : {sum(scores)/len(scores):.2f} / 5.0")
    print("------------------------------------------------------------------------\n")

def run_interactive_mode(use_llm=False):
    agent = AppleSupportAgent(use_llm=use_llm)
    b_trivial = BaselineTrivial()
    b_simple = BaselineSimple()

    print_banner()
    print("Type customer tweets below. Type 'exit' or 'quit' to exit.\n")

    sample_prompts = [
        "My Apple ID is locked and I can't receive my 2FA verification code!",
        "My iPhone 7 battery is swollen and screen is lifting up.",
        "How do I delete apps on iOS 11? 3D Touch pops up a menu.",
        "Keyboard lags terribly after iOS 11 update.",
        "Apple removing the headphone jack is pure greed!"
    ]
    print("Example prompts to try:")
    for idx, sp in enumerate(sample_prompts, 1):
        print(f"  {idx}. {sp}")
    print("\n" + "-"*72)

    while True:
        try:
            query = input("\n[Customer Tweet] > ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit"):
                print("Exiting interactive mode.")
                break

            res = agent.process(query, force_llm=use_llm)
            res_s = b_simple.process(query)
            res_t = b_trivial.process(query)

            print(f"\n--- [1] PROPOSED GROUNDED AGENT ---")
            print(f"• Intent:          {res['intent']} (Confidence: {res['intent_confidence']*100:.1f}%)")
            print(f"• Classification:  {res['intent_reason']}")
            status = "🚨 ESCALATE TO HUMAN / DM" if res['escalate'] else "✅ AUTO-HANDLE (SELF-SERVICE)"
            print(f"• Escalation:      {status}")
            print(f"• Policy Reason:   {res['escalation_reason']}")
            print(f"• Drafted Reply:   \"{res['drafted_reply']}\"")

            if res.get("retrieved_exemplars"):
                best = res["retrieved_exemplars"][0]
                print(f"• Historical Case: Matched Kaggle #{best.get('id', 'N/A')} (Score: {best.get('score', 0):.2f})")
                if best.get("kb_link"):
                    print(f"• Grounding Doc:   {best.get('kb_link')}")

            print(f"\n--- [2] BASELINE COMPARISON ---")
            print(f"• Baseline 1 (Trivial): Intent={res_t['intent']} | Escalate={res_t['escalate']}")
            print(f"• Baseline 2 (Simple):  Intent={res_s['intent']} | Escalate={res_s['escalate']}")
            print("-" * 72)
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

def run_tests():
    print("Running Python Unit Tests for AppleSupportAgent & Evaluation Harness...")
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(BASE_DIR, "tests"))
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)

def main():
    parser = argparse.ArgumentParser(description="@AppleSupport Grounded AI Customer Support Agent (Python)")
    parser.add_argument("--eval", action="store_true", help="Run full benchmark evaluation on 200 golden cases")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark on 50 sample cases")
    parser.add_argument("--query", type=str, help="Process a single incoming tweet")
    parser.add_argument("--interactive", action="store_true", help="Start interactive terminal testing REPL")
    parser.add_argument("--serve", action="store_true", help="Start pure Python HTTP server on specified port")
    parser.add_argument("--port", type=int, default=3000, help="Port for Python server (default: 3000)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind to")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--stats", action="store_true", help="Display golden dataset distribution stats")
    parser.add_argument("--llm", action="store_true", help="Enable live Gemini LLM reply drafting")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON for --query")

    args = parser.parse_args()

    # 1. Unit Tests
    if args.test:
        run_tests()
        return

    # 2. Dataset Stats
    if args.stats:
        show_dataset_stats()
        return

    # 3. Interactive REPL
    if args.interactive:
        run_interactive_mode(use_llm=args.llm)
        return

    # 4. Web Server
    if args.serve:
        from server import run_server
        run_server(host=args.host, port=args.port)
        return

    # 5. Single Query Inference
    if args.query:
        agent = AppleSupportAgent(use_llm=args.llm)
        b_trivial = BaselineTrivial()
        b_simple = BaselineSimple()

        res_agent = agent.process(args.query, force_llm=args.llm)
        res_t = b_trivial.process(args.query)
        res_s = b_simple.process(args.query)

        if args.json:
            print(json.dumps({
                "customer_text": args.query,
                "proposed_agent": res_agent,
                "baseline_trivial": res_t,
                "baseline_simple": res_s
            }, indent=2))
            return

        print_banner()
        print(f"Customer Tweet: \"{args.query}\"")
        print("\n[Intent Classification]:")
        print(f"  Intent:     {res_agent['intent']}")
        print(f"  Confidence: {res_agent['intent_confidence']*100:.1f}%")
        print(f"  Reason:     {res_agent['intent_reason']}")
        print("\n[Escalation Policy]:")
        print(f"  Escalate:   {'YES (Human Agent / DM Required)' if res_agent['escalate'] else 'NO (Auto-Handle)'}")
        print(f"  Tier:       {res_agent['escalation_policy_tier']}")
        print(f"  Reason:     {res_agent['escalation_reason']}")
        print("\n[Drafted Brand Reply]:")
        print(f"  \"{res_agent['drafted_reply']}\"")
        print("\n[Comparative Baselines]:")
        print(f"  Baseline 1 (Trivial): Intent={res_t['intent']}, Escalate={res_t['escalate']}")
        print(f"  Baseline 2 (Simple):  Intent={res_s['intent']}, Escalate={res_s['escalate']}")
        return

    # 6. Default: Benchmark Run
    print_banner()
    num_samples = 50 if args.quick else 200
    print(f"Executing Python Benchmark on {num_samples} Hand-Labelled Cases...")
    results = run_full_benchmark(sample_size=num_samples)
    print("\nBenchmark completed successfully in Python.")
    print("Run `python3 main.py --interactive` for live testing or `python3 main.py --serve` for web UI.")

if __name__ == "__main__":
    main()
