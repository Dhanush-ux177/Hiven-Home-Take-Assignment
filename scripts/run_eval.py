#!/usr/bin/env python3
"""
run_eval.py
CLI entrypoint to reproduce benchmark results in under 15 minutes (actually under 10 seconds!).
Usage:
  python3 scripts/run_eval.py             # Full benchmark across all 200 golden examples
  python3 scripts/run_eval.py --quick     # Fast benchmark sample (50 records)
  python3 scripts/run_eval.py --query "My Apple ID is locked!" # Test custom live query
"""

import sys
import os
import json
import argparse
from evaluator import run_full_benchmark, BenchmarkEvaluator
from agent import AppleSupportAgent
from baselines import BaselineTrivial, BaselineSimple

def main():
    parser = argparse.ArgumentParser(description="Evaluate Twitter Support AI Agent on Kaggle Golden Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run fast test on first 50 examples")
    parser.add_argument("--query", type=str, help="Run live classification & response on a custom customer tweet")
    parser.add_argument("--llm", action="store_true", help="Enable live Gemini LLM generation for replies")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")
    args = parser.parse_args()

    if args.query:
        agent = AppleSupportAgent(use_llm=args.llm)
        b_trivial = BaselineTrivial()
        b_simple = BaselineSimple()

        res = agent.process(args.query, force_llm=args.llm)
        res_t = b_trivial.process(args.query)
        res_s = b_simple.process(args.query)

        if args.json:
            out = {
                "customer_text": args.query,
                "proposed_agent": res,
                "baseline_trivial": res_t,
                "baseline_simple": res_s
            }
            print(json.dumps(out))
            return

        print(f"\n========================================================")
        print(f"       LIVE QUERY INFERENCE: @AppleSupport AGENT        ")
        print(f"========================================================")
        print(f"Incoming Customer Tweet: \"{args.query}\"")
        agent = AppleSupportAgent(use_llm=args.llm)
        res = agent.process(args.query, force_llm=args.llm)
        print(f"\n[1] Intent Classification:")
        print(f"    Intent:     {res['intent']}")
        print(f"    Confidence: {res['intent_confidence']*100:.1f}%")
        print(f"    Reason:     {res['intent_reason']}")
        print(f"\n[2] Escalation Policy Decision:")
        print(f"    Escalate:   {'YES (Human / DM Required)' if res['escalate'] else 'NO (Auto-Handle Self-Service)'}")
        print(f"    Policy Tier: {res['escalation_policy_tier']}")
        print(f"    Stated Reason: {res['escalation_reason']}")
        print(f"\n[3] Historical Grounding (Retrieved {len(res['retrieved_exemplars'])} Kaggle Cases):")
        for i, ex in enumerate(res['retrieved_exemplars'][:2]):
            print(f"    Case #{i+1} [{ex['intent']}]: \"{ex['query'][:70]}...\"")
        print(f"\n[4] Drafted Reply:")
        print(f"    \"{res['drafted_reply']}\"")
        print(f"========================================================\n")
        return

    print("Running @AppleSupport Customer Support Agent Benchmark...")
    if args.quick:
        evaluator = BenchmarkEvaluator()
        agent = AppleSupportAgent()
        res = evaluator.evaluate_model(agent, sample_limit=50)
        print(f"Quick evaluation complete on 50 samples.")
        print(f"Intent Accuracy: {res['intent_metrics']['accuracy']*100:.1f}%")
        print(f"Escalation F1:   {res['escalation_metrics']['f1']:.3f}")
        print(f"Safety (False Auto-Handle Rate): {res['escalation_metrics']['false_autohandle_rate']*100:.1f}%")
    else:
        run_full_benchmark()

if __name__ == "__main__":
    main()
