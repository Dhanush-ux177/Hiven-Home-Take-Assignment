#!/usr/bin/env python3
"""
Unit tests for @AppleSupport AI Agent, Baselines, and Evaluation Metrics.
Run with:
  python3 -m unittest discover tests
  or
  python3 main.py --test
"""

import os
import sys
import unittest

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
if SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, SCRIPTS_DIR)

from agent import AppleSupportAgent
from baselines import BaselineTrivial, BaselineSimple
from evaluator import BenchmarkEvaluator

class TestAppleSupportAgent(unittest.TestCase):
    def setUp(self):
        self.agent = AppleSupportAgent(use_llm=False)
        self.b_trivial = BaselineTrivial()
        self.b_simple = BaselineSimple()

    def test_account_security_escalation(self):
        """Account security and locked Apple IDs must trigger escalation for PII protection."""
        query = "My Apple ID is locked for security reasons and I can't receive 2FA code"
        res = self.agent.process(query)
        self.assertEqual(res["intent"], "ACCOUNT_SECURITY_ICLOUD")
        self.assertTrue(res["escalate"])
        self.assertEqual(res["escalation_policy_tier"], "CRITICAL_GUARDRAIL")

    def test_hardware_battery_hazard_escalation(self):
        """Swollen batteries and thermal hazards must escalate immediately for hardware safety."""
        query = "My battery is swollen and screen is lifting up! Is it dangerous?"
        res = self.agent.process(query)
        self.assertEqual(res["intent"], "HARDWARE_BATTERY")
        self.assertTrue(res["escalate"])
        self.assertEqual(res["escalation_policy_tier"], "CRITICAL_GUARDRAIL")

    def test_software_glitch_autohandle(self):
        """Simple software diagnostic questions should be auto-handled."""
        query = "Keyboard lag on iOS 11 when typing in Messages app on iPhone 6s"
        res = self.agent.process(query)
        self.assertEqual(res["intent"], "SOFTWARE_OS_BUG")
        self.assertFalse(res["escalate"])

    def test_device_setup_feature(self):
        """Feature discovery like 3D Touch app deletion should classify correctly."""
        query = "How do I delete apps on iPhone 7 with 3D touch? Quick actions menu keeps appearing."
        res = self.agent.process(query)
        self.assertEqual(res["intent"], "DEVICE_SETUP_FEATURE")

    def test_baselines_contract(self):
        """Ensure baselines return required fields."""
        query = "Battery draining fast"
        res_t = self.b_trivial.process(query)
        res_s = self.b_simple.process(query)
        
        self.assertIn("intent", res_t)
        self.assertIn("escalate", res_t)
        self.assertIn("intent", res_s)
        self.assertIn("escalate", res_s)
        self.assertFalse(res_t["escalate"]) # Trivial never escalates

    def test_historical_rag_retrieval(self):
        """BM25 historical retriever should find relevant authentic cases."""
        query = "Can't delete apps on home screen"
        res = self.agent.process(query)
        self.assertTrue(len(res["retrieved_exemplars"]) > 0)
        top_match = res["retrieved_exemplars"][0]
        self.assertIn("query", top_match)
        self.assertIn("reply", top_match)

if __name__ == "__main__":
    unittest.main()
