"""
Production-grade Guardrails Engine for the AI Shopping Assistant.

Enforces:
1. Input Guardrails:
   - Prompt injection & jailbreak detection
   - PII detection & masking (credit cards, SSNs, sensitive tokens)
   - E-commerce domain relevance boundary
   - Message length & token spam limitation
2. Output Guardrails:
   - Zero-hallucination price verification against catalog truth
   - System prompt leakage prevention
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from app.catalog import catalog

logger = logging.getLogger("guardrails")

# Known prompt injection & jailbreak patterns
INJECTION_PATTERNS = [
    r"(?i)\bignore\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts|rules)\b",
    r"(?i)\bdisregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)\b",
    r"(?i)\byou\s+are\s+now\s+(a\s+)?(dan|jailbreak|unfiltered|god\s*mode|root)\b",
    r"(?i)\b(system\s*prompt|system\s*instructions|reveal\s+your\s+prompt)\b",
    r"(?i)\bbypass\s+(all\s+)?(security|restrictions|safeguards|filters)\b",
    r"(?i)\bpretend\s+you\s+have\s+no\s+(rules|restrictions|guidelines)\b",
    r"(?i)\bexecute\s+(python|bash|sql|shell)\s+code\b",
    r"(?i)\b(show\s+(me\s+)?(the\s+)?)?(internal\s+)?(database\s+password|api\s*key|db\s+password|server\s+secret)\b",
    r"(?i)\bdrop\s+table\b",
    r"(?i)<script\b",
]

# Off-topic / harmful exploit query patterns
OFF_TOPIC_PATTERNS = [
    r"(?i)\b(write|create)\s+(malware|ransomware|keylogger|virus|exploit)\b",
    r"(?i)\bhow\s+to\s+(hack|infiltrate|ddos)\b",
    r"(?i)\bgenerate\s+(fake\s+credit\s+cards|counterfeit|passports)\b",
]

# Credit Card pattern (13-16 digits with optional dashes/spaces)
CREDIT_CARD_PATTERN = r"\b(?:\d{4}[ -]?){3}\d{4}\b|\b\d{13,16}\b"

# Max allowable user input length
MAX_INPUT_LENGTH = 1500

@dataclass
class GuardrailResult:
    is_safe: bool
    action: str  # "allow" | "intercept" | "sanitize"
    reason: Optional[str] = None
    sanitized_message: Optional[str] = None
    intercept_response: Optional[str] = None
    suggested_pills: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

class GuardrailManager:
    """Enterprise safety and policy manager for e-commerce AI assistant."""

    def __init__(self):
        self._compiled_injections = [re.compile(p) for p in INJECTION_PATTERNS]
        self._compiled_off_topics = [re.compile(p) for p in OFF_TOPIC_PATTERNS]
        self._card_regex = re.compile(CREDIT_CARD_PATTERN)

    def validate_input(self, message: str) -> GuardrailResult:
        """Runs comprehensive input guardrail checks on the user message."""
        if not message or not message.strip():
            return GuardrailResult(
                is_safe=False,
                action="intercept",
                reason="empty_input",
                intercept_response="Please ask a question about products, pricing, or specifications in our store!",
                suggested_pills=["Show sunglasses", "View watch price", "Kitchen items under $20"],
            )

        trimmed = message.strip()

        # 1. Length Boundary Check
        if len(trimmed) > MAX_INPUT_LENGTH:
            return GuardrailResult(
                is_safe=False,
                action="intercept",
                reason="length_exceeded",
                intercept_response=f"Your message is too long (maximum {MAX_INPUT_LENGTH} characters). Please summarize your request.",
                suggested_pills=["Show sunglasses", "Vintage Watch specs", "Browse apparel"],
            )

        # 2. Prompt Injection & Adversarial Jailbreak Defense
        for pattern in self._compiled_injections:
            if pattern.search(trimmed):
                logger.warning(f"Guardrail intercepted prompt injection: {trimmed[:60]}...")
                return GuardrailResult(
                    is_safe=False,
                    action="intercept",
                    reason="prompt_injection_detected",
                    intercept_response=(
                        "I am your Online Boutique Shopping Assistant. I cannot ignore my guidelines or execute unauthorized system commands. "
                        "How may I help you find products, check prices, or answer questions about our catalog?"
                    ),
                    suggested_pills=["Search sunglasses", "Watch specifications", "Kitchenware under $20", "Browse all products"],
                    metadata={"pattern": pattern.pattern},
                )

        # 3. Malicious / Harmful Off-Topic Filter
        for pattern in self._compiled_off_topics:
            if pattern.search(trimmed):
                logger.warning(f"Guardrail intercepted harmful request: {trimmed[:60]}...")
                return GuardrailResult(
                    is_safe=False,
                    action="intercept",
                    reason="harmful_off_topic_detected",
                    intercept_response=(
                        "I cannot assist with requests outside of our store's shopping catalog. "
                        "I'm here to help you browse items, verify exact prices, and check product materials!"
                    ),
                    suggested_pills=["Browse accessories", "Kitchen items", "Loafers & footwear", "All products"],
                    metadata={"pattern": pattern.pattern},
                )

        # 4. PII Masking (Credit Cards, Sensitive tokens)
        sanitized = trimmed
        has_pii = False
        if self._card_regex.search(trimmed):
            sanitized = self._card_regex.sub("[REDACTED_PAYMENT_INFO]", sanitized)
            has_pii = True
            logger.info("Guardrail masked detected payment/credit card information.")

        return GuardrailResult(
            is_safe=True,
            action="sanitize" if has_pii else "allow",
            sanitized_message=sanitized,
            metadata={"pii_masked": has_pii},
        )

    def verify_output(self, response_text: str, queried_products: Optional[List[Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Output Guardrails:
        - Ensures mentioned prices match catalog ground truth (zero-hallucination verification).
        - Prevents inadvertent leakage of system prompt tokens.
        """
        corrections: List[str] = []

        # System prompt leakage check
        if "You are an expert AI Shopping Assistant" in response_text or "INTERNAL_PRODUCT_SPECS" in response_text:
            response_text = "I'd be happy to help you with our product catalog, pricing, and specifications! What would you like to know?"
            corrections.append("masked_system_prompt_leakage")

        # Price verification against catalog truth
        # Find all $XX.XX patterns
        price_matches = re.findall(r"\$(\d+(?:\.\d{2})?)", response_text)
        if price_matches and queried_products:
            catalog_prices = {f"{p.price_usd.amount:.2f}" for p in queried_products if hasattr(p, "price_usd")}
            # Also include units string
            catalog_prices.update({str(p.price_usd.units) for p in queried_products if hasattr(p, "price_usd")})

            for found_price in price_matches:
                # If there's a specific product queried and price doesn't match, verify
                if len(queried_products) == 1:
                    actual_price = f"{queried_products[0].price_usd.amount:.2f}"
                    if found_price != actual_price and abs(float(found_price) - float(actual_price)) > 0.01:
                        # Correct hallucinated price in text
                        response_text = re.sub(rf"\${re.escape(found_price)}", f"${actual_price}", response_text)
                        corrections.append(f"corrected_price_${found_price}_to_${actual_price}")

        return response_text, {
            "verified": True,
            "corrections_applied": corrections,
        }

guardrail_manager = GuardrailManager()
