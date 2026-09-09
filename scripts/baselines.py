#!/usr/bin/env python3
"""
baselines.py
Baseline systems for comparative benchmarking:
1. BaselineTrivial (Majority class intent, fixed canned reply, zero escalation)
2. BaselineSimple (Naive ungrounded keyword matching, length-based heuristic escalation, generic prompt reply)
"""

import re
from typing import Dict, Any, List

class BaselineTrivial:
    """Trivial Baseline:
    - Intent: Always predicts majority class ('SOFTWARE_OS_BUG').
    - Reply: Static generic boilerplate.
    - Escalation: Always auto-handle (False) without reasoning.
    """
    def __init__(self):
        self.name = "Baseline 1: Trivial (Majority Class + Canned)"
        self.canned_reply = (
            "Hello! Thanks for reaching out to Apple Support. Please restart your device "
            "and visit support.apple.com for more help. Let us know if you need anything else!"
        )

    def process(self, customer_text: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        return {
            "intent": "SOFTWARE_OS_BUG",
            "intent_confidence": 0.50,
            "intent_reason": "Default majority class assignment",
            "escalate": False,
            "escalation_confidence": 0.50,
            "escalation_policy_tier": "TRIVIAL_DEFAULT",
            "escalation_reason": "Static default: all messages routed to automated canned reply",
            "drafted_reply": self.canned_reply,
            "retrieved_exemplars": []
        }

class BaselineSimple:
    """Simple Baseline:
    - Intent: Naive keyword matching without confidence weighting or multi-intent disambiguation.
    - Escalation: Superficial heuristic based on punctuation/length (e.g. >2 exclamation marks or length > 180 chars).
    - Reply: Ungrounded template reply without historical brand exemplars or official HT resolution links.
    """
    def __init__(self):
        self.name = "Baseline 2: Simple (Naive Keyword + Length Escalation)"

    def process(self, customer_text: str, conversation_history: List[str] = None) -> Dict[str, Any]:
        text_lower = customer_text.lower()
        
        # Naive first-hit keyword mapping (prone to error on compound queries)
        if "apple id" in text_lower or "password" in text_lower or "charge" in text_lower:
            intent = "ACCOUNT_SECURITY_ICLOUD"
        elif "battery" in text_lower or "screen" in text_lower:
            intent = "HARDWARE_BATTERY"
        elif "store" in text_lower or "order" in text_lower:
            intent = "STORE_REPAIR_ORDER"
        elif "how to" in text_lower or "how do" in text_lower:
            intent = "DEVICE_SETUP_FEATURE"
        elif "hate" in text_lower or "worst" in text_lower or "sucks" in text_lower:
            intent = "GENERAL_FEEDBACK_RANT"
        else:
            intent = "SOFTWARE_OS_BUG"

        # Naive escalation: only escalates if customer used angry exclamation marks or wrote a long tweet
        exclamation_count = customer_text.count("!")
        is_long = len(customer_text) > 160
        escalate = (exclamation_count >= 2) or is_long
        
        reason = (
            "Escalated due to high punctuation/message length threshold"
            if escalate
            else "Auto-handled as message does not meet punctuation length trigger"
        )

        # Naive ungrounded reply
        if escalate:
            reply = "We notice you seem upset. Please contact an agent or DM us for further assistance."
        else:
            reply = f"We see you have a question about your device ({intent}). Please visit apple.com or try restarting."

        return {
            "intent": intent,
            "intent_confidence": 0.70,
            "intent_reason": f"First keyword match for {intent}",
            "escalate": escalate,
            "escalation_confidence": 0.65,
            "escalation_policy_tier": "NAIVE_HEURISTIC",
            "escalation_reason": reason,
            "drafted_reply": reply,
            "retrieved_exemplars": []
        }
