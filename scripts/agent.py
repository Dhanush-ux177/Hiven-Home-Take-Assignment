#!/usr/bin/env python3
"""
agent.py
Grounded AI Customer Support Agent for @AppleSupport.

Components:
1. IntentClassifier (Calibrated 6-class intent taxonomy)
2. EscalationEngine (Domain safety & operational escalation policy)
3. HistoricalRetriever (BM25 / TF-IDF retrieval over Kaggle historical exemplars)
4. ReplyDrafter (Brand-faithful grounded drafting with official KB anchors)
"""

import json
import os
import re
import math
import urllib.request
from collections import Counter
from typing import Dict, List, Any, Optional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
EXEMPLARS_FILE = os.path.join(DATA_DIR, "historical_exemplars.json")

INTENTS = [
    "SOFTWARE_OS_BUG",
    "HARDWARE_BATTERY",
    "ACCOUNT_SECURITY_ICLOUD",
    "DEVICE_SETUP_FEATURE",
    "STORE_REPAIR_ORDER",
    "GENERAL_FEEDBACK_RANT"
]

class HistoricalRetriever:
    """Retrieves historically resolved Apple Support interactions from Kaggle dataset."""
    def __init__(self, exemplars_path: str = EXEMPLARS_FILE):
        self.exemplars = []
        if os.path.exists(exemplars_path):
            with open(exemplars_path, "r", encoding="utf-8") as f:
                self.exemplars = json.load(f)
        else:
            self.exemplars = []
            
        # Build inverted index for fast keyword matching / BM25
        self.doc_tokens = []
        self.vocab = {}
        self.df = Counter()
        self.N = len(self.exemplars)
        
        for i, doc in enumerate(self.exemplars):
            text = (doc.get("customer_query", "") + " " + doc.get("intent", "")).lower()
            tokens = re.findall(r'\b\w{2,}\b', text)
            self.doc_tokens.append(tokens)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                self.df[t] += 1
                if t not in self.vocab:
                    self.vocab[t] = len(self.vocab)
                    
        self.avg_doc_len = sum(len(d) for d in self.doc_tokens) / max(1, self.N)

    def retrieve(self, query: str, intent_hint: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        query_tokens = re.findall(r'\b\w{2,}\b', query.lower())
        if not query_tokens:
            return self.exemplars[:top_k]

        scores = []
        k1 = 1.5
        b = 0.75
        
        for idx, doc in enumerate(self.exemplars):
            doc_len = len(self.doc_tokens[idx])
            score = 0.0
            doc_term_freq = Counter(self.doc_tokens[idx])
            
            for qt in query_tokens:
                if qt in self.df:
                    n_qt = self.df[qt]
                    idf = math.log((self.N - n_qt + 0.5) / (n_qt + 0.5) + 1.0)
                    tf = doc_term_freq[qt]
                    numerator = tf * (k1 + 1.0)
                    denominator = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (numerator / max(1e-6, denominator))
            
            # Boost if intent matches
            if intent_hint and doc.get("intent") == intent_hint:
                score *= 1.4

            scores.append((score, doc))

        scores.sort(key=lambda x: x[0], reverse=True)
        results = []
        for s, doc in scores[:top_k]:
            results.append({
                "id": doc.get("id"),
                "intent": doc.get("intent"),
                "query": doc.get("customer_query"),
                "reply": doc.get("resolved_reply"),
                "action": doc.get("recommended_action"),
                "kb_link": doc.get("kb_link", "https://support.apple.com"),
                "score": round(s, 3)
            })
        return results

class IntentClassifier:
    """Classifies customer message into 6 operational support intents."""
    def __init__(self):
        self.keyword_rules = {
            "ACCOUNT_SECURITY_ICLOUD": [
                (r'\b(apple id|icloud|locked|lockout|password|passcode|unauthorized|stolen|hacked|phishing|2fa|two-factor|verification code|billing|refund|charged|subscription|purchase history|cant log in|can\'t log in|id disabled)\b', 3.0),
                (r'\b(receipt|credit card|applecare billing|compromised|reset password)\b', 2.0)
            ],
            "HARDWARE_BATTERY": [
                (r'\b(battery|swollen|swelling|overheat|overheating|hot|crack|cracked|shattered|broken screen|screen lift|screen pop|hardware|speaker crackle|microphone dead|lightning port|charging port|water damage|dropped)\b', 3.0),
                (r'\b(drain|draining|dies fast|battery health|shut down|shuts down|wont charge|won\'t charge|charger broken)\b', 2.2)
            ],
            "STORE_REPAIR_ORDER": [
                (r'\b(genius bar|store appointment|apple store appointment|repair status|trade-in kit|trade in|order status|shipment|delivery date|pickup in store|walk in|warranty replacement)\b', 3.2),
                (r'\b(order number|shipping|courier|ups|fedex|repair id)\b', 2.0)
            ],
            "DEVICE_SETUP_FEATURE": [
                (r'\b(how do i|how to|how can i|where is|delete apps|wiggle|airpods pair|pairing|family sharing|family share|airdrop|transfer data|move to ios|screen recording|carplay setup|customize controls|wishlist)\b', 2.8),
                (r'\b(settings|feature|bluetooth connect|switch on|configure|instructions)\b', 1.8)
            ],
            "GENERAL_FEEDBACK_RANT": [
                (r'\b(sucks|worst update|fuq|fuck|shitphone|garbage|trash|hate|greed|corporate greed|steve jobs|ridiculous|disappointed|shame on you|terrible company|joke|cagou)\b', 3.0),
                (r'\b(why did you remove|bring back|ruined my phone|unusable)\b', 2.0)
            ],
            "SOFTWARE_OS_BUG": [
                (r'\b(ios 11|ios 10|update bug|glitch|freeze|freezing|crashes|crashing|lag|laggy|keyboard lag|bluetooth turning on|wifi dropping|wi-fi drop|podcast stop|music disappeared|voice control bug|safari crash|reboot)\b', 2.5),
                (r'\b(bug|buggy|issue|slow|latency|restart|apple music app|syncing stuck)\b', 1.5)
            ]
        }

    def classify(self, text: str) -> Dict[str, Any]:
        text_lower = text.lower()
        scores = {intent: 0.1 for intent in INTENTS}
        matches = {intent: [] for intent in INTENTS}

        for intent, patterns in self.keyword_rules.items():
            for pattern, weight in patterns:
                found = re.findall(pattern, text_lower)
                if found:
                    scores[intent] += weight * len(found)
                    matches[intent].extend(found)

        # Softmax / normalize confidence
        max_score = max(scores.values())
        exp_scores = {k: math.exp(v - max_score) for k, v in scores.items()}
        total_exp = sum(exp_scores.values())
        probs = {k: round(v / total_exp, 3) for k, v in exp_scores.items()}

        best_intent = max(probs.items(), key=lambda x: x[1])[0]
        confidence = probs[best_intent]

        # Rationales
        matched_tokens = matches[best_intent][:3]
        if matched_tokens:
            reason = f"Detected high-salience terms: {', '.join(set(matched_tokens))}"
        else:
            reason = "General system inquiry mapped to closest operational category"

        return {
            "intent": best_intent,
            "confidence": confidence,
            "probabilities": probs,
            "reason": reason
        }

class EscalationEngine:
    """Decides whether to auto-handle or escalate to human technician / secure DM."""
    def __init__(self):
        self.critical_escalation_patterns = [
            # Security & PII triggers
            (r'\b(apple id|locked|password|passcode|unauthorized|2fa|verification code|hacked|billing charge|fraud|stolen|phishing)\b',
             "Requires private authentication & credential protection; PII cannot be handled in public Twitter tweets"),
            # Hardware safety & diagnostic triggers
            (r'\b(swollen|swelling|overheat|hot to touch|crack|cracked|shattered|screen broke|water damage|spark|fire)\b',
             "Physical hardware damage or safety hazard requires Genius Bar physical inspection or Apple Authorized repair"),
            # Logistics & store triggers
            (r'\b(genius bar appointment|repair status|trade-in kit|order status|shipment delayed|pickup)\b',
             "Localized store booking or internal order shipment lookup requires private customer account access"),
            # Legal / Churn / Executive Escalation
            (r'\b(lawsuit|lawyer|better business bureau|bbb|attorney|compensation|executive relations)\b',
             "Legal dispute or high-severity customer grievance requires senior human escalation manager")
        ]

    def decide(self, text: str, intent_info: Dict[str, Any], thread_depth: int = 1) -> Dict[str, Any]:
        text_lower = text.lower()
        intent = intent_info.get("intent", "")
        
        # 1. Check critical safety/security patterns
        for pat, reason in self.critical_escalation_patterns:
            if re.search(pat, text_lower):
                return {
                    "escalate": True,
                    "confidence": 0.96,
                    "policy_tier": "CRITICAL_GUARDRAIL",
                    "reason": reason
                }

        # 2. Intent-based baseline policy
        if intent == "ACCOUNT_SECURITY_ICLOUD":
            return {
                "escalate": True,
                "confidence": 0.92,
                "policy_tier": "INTENT_SECURITY_POLICY",
                "reason": "Account access and billing disputes require secure DM or iforgot.apple.com authentication"
            }

        if intent == "STORE_REPAIR_ORDER":
            return {
                "escalate": True,
                "confidence": 0.88,
                "policy_tier": "INTENT_STORE_POLICY",
                "reason": "Store appointment scheduling and order tracking require localized reservation system lookup"
            }

        if intent == "HARDWARE_BATTERY":
            # If purely software battery drain vs physical defect
            if any(w in text_lower for w in ["drain", "draining", "battery life", "die fast", "percent"]):
                return {
                    "escalate": False,
                    "confidence": 0.85,
                    "policy_tier": "SELF_SERVICE_BATTERY_POLICY",
                    "reason": "Battery consumption and optimization can be resolved via iOS Settings & Battery Health self-service"
                }
            return {
                "escalate": True,
                "confidence": 0.89,
                "policy_tier": "HARDWARE_DIAGNOSTIC_POLICY",
                "reason": "Hardware component triage requires remote diagnostic session in DM or Genius Bar booking"
            }

        # 3. Thread depth / repeated frustration
        if thread_depth >= 4:
            return {
                "escalate": True,
                "confidence": 0.82,
                "policy_tier": "MULTI_TURN_EXHAUSTION_POLICY",
                "reason": "Customer inquiry has exceeded 3 turns without resolution; escalate to dedicated human support specialist"
            }

        # 4. Auto-handle: Software OS bug, How-To, General Feedback
        if intent == "DEVICE_SETUP_FEATURE":
            return {
                "escalate": False,
                "confidence": 0.94,
                "policy_tier": "SELF_SERVICE_KNOWLEDGE_POLICY",
                "reason": "Standard device feature instruction covered by public knowledge base documentation"
            }

        if intent == "GENERAL_FEEDBACK_RANT":
            return {
                "escalate": False,
                "confidence": 0.90,
                "policy_tier": "BRAND_DEESCALATION_POLICY",
                "reason": "Sentiment logged for product feedback; no technical account intervention required"
            }

        # Default for SOFTWARE_OS_BUG
        return {
            "escalate": False,
            "confidence": 0.84,
            "policy_tier": "DIAGNOSTIC_ISOLATION_POLICY",
            "reason": "Standard software triage (device model check, OS version, isolation restart) applies"
        }

class ReplyDrafter:
    """Synthesizes brand-faithful replies grounded in Apple's customer support communications."""
    def __init__(self, api_key: Optional[str] = None, use_llm: bool = False):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.use_llm = use_llm

    def draft(
        self,
        customer_text: str,
        intent_info: Dict[str, Any],
        escalation_info: Dict[str, Any],
        exemplars: List[Dict[str, Any]],
        conversation_history: List[str] = None,
        force_llm: bool = False
    ) -> str:
        intent = intent_info.get("intent", "SOFTWARE_OS_BUG")
        escalate = escalation_info.get("escalate", False)
        escalation_reason = escalation_info.get("reason", "")
        top_ex = exemplars[0] if exemplars else {}
        kb_link = top_ex.get("kb_link", "https://support.apple.com")

        # Try calling Gemini API if requested and key is available
        if (self.use_llm or force_llm) and self.api_key:
            reply = self._draft_with_gemini(
                customer_text, intent, escalate, escalation_reason, exemplars, conversation_history
            )
            if reply:
                return reply

        # Fast, deterministic grounded synthesis (sub-millisecond, highly calibrated)
        return self._draft_grounded_template(customer_text, intent, escalate, escalation_reason, top_ex)

    def _draft_with_gemini(
        self,
        customer_text: str,
        intent: str,
        escalate: bool,
        escalation_reason: str,
        exemplars: List[Dict[str, Any]],
        conversation_history: List[str] = None
    ) -> Optional[str]:
        try:
            exemplars_text = "\n\n".join([
                f"Example #{i+1} (Intent: {e.get('intent')}):\nCustomer: {e.get('query')}\nSupport Reply: {e.get('reply')}\nGuidance: {e.get('action')}"
                for i, e in enumerate(exemplars[:2])
            ])
            
            prompt = f"""You are the official Twitter support agent for @AppleSupport.
Write a brand-compliant, helpful, empathetic, concise tweet reply (under 280 characters if possible, maximum 2 short sentences).

CONTEXT:
Customer Tweet: "{customer_text}"
Classified Intent: {intent}
Escalation Status: {"ESCALATE TO HUMAN / DM" if escalate else "AUTO-HANDLE (SELF-SERVICE)"}
Escalation Reason: {escalation_reason}

HISTORICAL RESOLUTIONS FROM @AppleSupport:
{exemplars_text}

BRAND COMMUNICATION RULES:
1. Tone: Empathetic, calm, professional, and direct. Use phrasing like "We'd love to look into this with you", "Let's work together", or "We understand how concerning this is."
2. Never ask for passwords, Apple ID credentials, or credit card numbers on public Twitter.
3. If Escalation is required: Kindly invite them to DM ("Please send us a DM with your device details and we'll take it from there.") or direct to iforgot.apple.com / Genius Bar.
4. If Auto-Handle: Ask a focused diagnostic isolation question (e.g. "Which device model and iOS version are you currently running?") or provide standard self-service instructions.
5. Do NOT use emojis excessively (maximum 1 or none).

Generate ONLY the tweet response text."""

            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.6-flash:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.2,
                    "maxOutputTokens": 150
                }
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                res = json.loads(r.read().decode())
                text = res["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Clean quotes
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1].strip()
                return text
        except Exception as e:
            # Silently fall back to grounded template
            return None

    def _draft_grounded_template(
        self,
        customer_text: str,
        intent: str,
        escalate: bool,
        escalation_reason: str,
        top_ex: Dict[str, Any]
    ) -> str:
        text_lower = customer_text.lower()

        if intent == "ACCOUNT_SECURITY_ICLOUD":
            if "locked" in text_lower or "disabled" in text_lower or "cant log in" in text_lower:
                return "We know how important your Apple ID is. To safely unlock your account without sharing personal info publicly, visit https://iforgot.apple.com or DM us with any questions."
            return "We take account security very seriously. To protect your data, please never post credentials publicly. Send us a DM and we'll guide you through securing your account."

        if intent == "HARDWARE_BATTERY":
            if any(w in text_lower for w in ["swollen", "crack", "shatter", "broke", "water", "overheat"]):
                return "Your safety is our top priority. Please stop charging the device immediately. We recommend booking an inspection at your nearest Apple Store: https://getsupport.apple.com"
            return "Let's check your battery performance together. Head to Settings > Battery > Battery Health to check your Maximum Capacity, or DM us so we can run a remote diagnostic."

        if intent == "STORE_REPAIR_ORDER":
            return "We'd be glad to help with your store appointment or order. Please meet us in DM with your order details so we can look into this safely: https://getsupport.apple.com"

        if intent == "DEVICE_SETUP_FEATURE":
            if "delete apps" in text_lower or "wiggle" in text_lower:
                return "With 3D Touch, gently touch and rest your finger on the app icon without pressing down hard into the screen. Wait a moment until the apps begin to wiggle."
            if "airpods" in text_lower:
                return "Let's get your AirPods connected! Open the lid with AirPods inside, hold the button on the back until the light flashes amber then white, and hold near your device."
            return "We're happy to guide you through this setup! You can find full step-by-step instructions in our support guide: https://support.apple.com"

        if intent == "GENERAL_FEEDBACK_RANT":
            return "We hear you, and we appreciate you taking the time to share your perspective. You can submit your direct product suggestions to our engineering teams at https://apple.com/feedback"

        # SOFTWARE_OS_BUG default
        if escalate:
            return "We'd love to help get this sorted out. Which device model and software version are you using? Please send us a DM so we can troubleshoot together."
        return "We can help look into this. Which device model and iOS version are you currently running? Also, does this persist after a quick restart?"

class AppleSupportAgent:
    """End-to-end Grounded AI Customer Support Agent for @AppleSupport."""
    def __init__(self, api_key: Optional[str] = None, use_llm: bool = False):
        self.retriever = HistoricalRetriever()
        self.classifier = IntentClassifier()
        self.escalator = EscalationEngine()
        self.drafter = ReplyDrafter(api_key=api_key, use_llm=use_llm)

    def process(self, customer_text: str, conversation_history: List[str] = None, force_llm: bool = False) -> Dict[str, Any]:
        # 1. Intent Classification
        intent_info = self.classifier.classify(customer_text)
        
        # 2. Historical Retrieval (RAG)
        exemplars = self.retriever.retrieve(customer_text, intent_hint=intent_info["intent"], top_k=3)
        
        # 3. Escalation Decision
        depth = len(conversation_history) if conversation_history else 1
        escalation_info = self.escalator.decide(customer_text, intent_info, thread_depth=depth)
        
        # 4. Grounded Reply Drafting
        reply = self.drafter.draft(
            customer_text, intent_info, escalation_info, exemplars, conversation_history, force_llm=force_llm
        )

        return {
            "intent": intent_info["intent"],
            "intent_confidence": intent_info["confidence"],
            "intent_reason": intent_info["reason"],
            "probabilities": intent_info.get("probabilities", {}),
            "escalate": escalation_info["escalate"],
            "escalation_confidence": escalation_info["confidence"],
            "escalation_policy_tier": escalation_info.get("policy_tier", "DEFAULT"),
            "escalation_reason": escalation_info["reason"],
            "drafted_reply": reply,
            "retrieved_exemplars": exemplars
        }

if __name__ == "__main__":
    agent = AppleSupportAgent()
    test_queries = [
        "My iPhone 7 battery is draining in 2 hours since iOS 11. What can I do?",
        "My Apple ID has been locked for security reasons and I can't get verification code!",
        "My battery is swollen and screen is popping out! Is it dangerous?",
        "How do I delete apps if pressing down just opens quick actions?",
        "Removing the headphone jack is pure greed. Apple sucks now."
    ]
    for q in test_queries:
        res = agent.process(q)
        print(f"\n--- Customer: {q}")
        print(f"Intent: {res['intent']} ({res['intent_confidence']:.2f})")
        print(f"Escalate: {res['escalate']} ({res['escalation_reason']})")
        print(f"Reply: {res['drafted_reply']}")
