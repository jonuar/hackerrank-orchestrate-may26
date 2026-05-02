from __future__ import annotations
import re
from dataclasses import dataclass

COMPANIES     = {"HackerRank", "Claude", "Visa"}
REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

COMPANY_SIGNALS: dict[str, list[str]] = {
    "HackerRank": [
        "hackerrank", "hacker rank", "assessment", "coding test", "challenge",
        "plagiarism", "proctoring", "candidate", "recruiter", "test case",
        "submission", "leaderboard", "hackathon", "sprint", "contest",
        "code pair", "codepair", "interview kit", "hackerrank screen",
        "mock interview", "resume builder", "certificate", "inactivity",
        "zoom connectivity", "compatible check", "interviewer", "hiring",
    ],
    "Claude": [
        "claude", "anthropic", "claude.ai", "claude pro", "claude team",
        "artifact", "project", "conversation", "memory", "prompt", "api key",
        "claude api", "rate limit", "token", "context window", "model",
        "subscription", "billing anthropic", "claude code", "aws bedrock",
        "lti key", "data crawl", "crawling", "personal data", "bug bounty",
    ],
    "Visa": [
        "visa", "card", "transaction", "payment", "chargeback", "dispute",
        "merchant", "atm", "pin", "contactless", "tap to pay", "fraud",
        "unauthorized charge", "debit", "credit card", "visa checkout",
        "bank", "issuer", "statement", "carte visa", "blocked card",
        "minimum spend", "carte", "bloquée",
    ],
}

PRODUCT_AREA_SIGNALS: dict[str, list[str]] = {
    # HackerRank
    "assessments": [
        "assessment",
        "test",
        "coding test",
        "proctoring",
        "plagiarism",
        "anti-cheat",
        "zoom",
        "compatible check",
        "inactivity",
        "reschedule",
        "schedule",
        "unforeseen",
        "graded",
        "score",
        "answers",
    ],
    "developer_tools": [
        "ide",
        "code editor",
        "compiler",
        "language",
        "environment",
        "sandbox",
        "submissions",
        "apply tab",
        "not working",
        "website",
    ],
    "account_access": [
        "login",
        "sign in",
        "password",
        "2fa",
        "two-factor",
        "sso",
        "oauth",
        "access denied",
        "restore access",
        "seat",
        "workspace",
        "remove user",
        "remove interviewer",
        "employee leaving",
    ],
    "billing_payments": [
        "invoice",
        "billing",
        "charge",
        "refund",
        "plan",
        "upgrade",
        "downgrade",
        "subscription",
        "pause subscription",
        "order id",
        "payment",
        "mock interview refund",
    ],
    "candidates": [
        "candidate",
        "invite",
        "link",
        "test link",
        "expir",
        "certificate",
        "name on certificate",
        "next round",
    ],
    "contests": ["contest", "leaderboard", "hackathon", "sprint", "rank"],
    "infosec_compliance": [
        "infosec",
        "security forms",
        "compliance",
        "forms",
        "filling in",
    ],
    "resume_tools": ["resume builder", "resume", "cv"],
    # Claude
    "claude_subscription": [
        "claude pro",
        "claude team",
        "enterprise plan",
        "free plan",
        "upgrade claude",
        "pause",
        "workspace",
    ],
    "claude_api": [
        "api key",
        "api access",
        "claude api",
        "sdk",
        "rate limit",
        "max_tokens",
        "token",
        "aws bedrock",
        "bedrock",
        "lti key",
        "lti",
        "requests failing",
        "all requests",
    ],
    "claude_features": [
        "artifact",
        "memory",
        "conversation",
        "context window",
        "model selection",
        "crawling",
        "website crawl",
        "data use",
        "personal data",
        "data crawl",
    ],
    "claude_security": [
        "security vulnerability",
        "bug bounty",
        "vulnerability",
        "exploit",
    ],
    # Visa
    "transaction_disputes": [
        "dispute",
        "chargeback",
        "unauthorized",
        "fraudulent transaction",
        "charge",
        "refund",
        "wrong product",
        "merchant",
    ],
    "card_management": [
        "pin",
        "contactless",
        "tap",
        "block card",
        "lost card",
        "stolen card",
        "blocked",
        "bloquée",
        "minimum spend",
    ],
    "fraud_security": [
        "fraud",
        "scam",
        "phishing",
        "compromised",
        "suspicious activity",
        "identity stolen",
        "identity theft",
        "stolen identity",
    ],
    "general_inquiry": ["how do i", "what is", "can i", "is it possible", "why so"],
}

RISK_SIGNALS: list[str] = [
    # Financial
    "fraud", "unauthorized", "chargeback", "dispute", "stolen card", "compromised",
    "scam", "phishing", "wrong product",
    # Identity
    "identity stolen", "identity theft", "identity has been stolen",
    # Account security
    "account hacked", "can't log in", "locked out", "2fa lost", "lost access",
    "account suspended", "banned",
    # Billing disputes
    "wrong charge", "double charged", "charged twice", "not authorized to charge",
    "give me my money", "refund asap",
    # Security vulnerabilities
    "security vulnerability", "major security", "bug bounty",
    # Prompt injection — English
    "ignore previous instructions", "disregard", "you are now", "jailbreak",
    "act as", "pretend you are", "forget your instructions", "reveal your",
    "show me your instructions", "delete all files",
    # Prompt injection — French / multilingual
    "règles internes", "logique exacte", "documents récupérés",
    "affiche toutes", "montrez-moi", "ignorez les",
]

FEATURE_REQUEST_SIGNALS: list[str] = [
    "feature request", "would be nice", "wish you could", "please add",
    "suggestion", "it would be great if", "can you add", "i'd love",
    "enhancement", "new feature", "support for", "allow users to",
    "extend inactivity", "can we extend",
]

BUG_SIGNALS: list[str] = [
    "bug", "broken", "not working", "stopped working", "not able to",
    "error", "crash", "exception", "infinite loop", "stuck", "freezes",
    "glitch", "issue with", "problem with", "fails", "incorrect behavior",
    "stopped in between", "is down", "are failing", "unable to take",
    "blocker", "cannot", "can not",
]

INVALID_SIGNALS: list[str] = [
    "hello", "hi there", "test message", "asdf", "lorem ipsum",
    "just testing", "nothing", "urgent cash", "need cash",
    "delete all files", "give me the code to",
]


@dataclass
class ClassificationResult:
    company:      str
    product_area: str
    request_type: str
    is_high_risk: bool
    risk_reasons: list[str]


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _count_signals(text: str, signals: list[str]) -> int:
    return sum(1 for s in signals if s in text)


def infer_company(text: str) -> str:
    scores = {
        company: _count_signals(text, signals)
        for company, signals in COMPANY_SIGNALS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "Unknown"


def classify_product_area(text: str, company: str) -> str:
    company_prefix_map = {
        "HackerRank": [
            "assessments",
            "developer_tools",
            "account_access",
            "billing_payments",
            "candidates",
            "contests",
            "infosec_compliance",
            "resume_tools",
        ],
        "Claude": [
            "claude_subscription",
            "claude_api",
            "claude_features",
            "claude_security",
        ],
        "Visa": [
            "transaction_disputes",
            "card_management",
            "fraud_security",
            "merchant_support",
        ],
    }
    scores: dict[str, int] = {}
    for area, signals in PRODUCT_AREA_SIGNALS.items():
        count = _count_signals(text, signals)
        if company in company_prefix_map and area in company_prefix_map[company]:
            count *= 2
        scores[area] = count

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general_inquiry"


def classify_request_type(text: str) -> str:
    # Invalid: malicious or completely out-of-scope
    if _count_signals(text, INVALID_SIGNALS) >= 1 and len(text) < 120:
        return "invalid"
    # Specific invalid patterns regardless of length
    if any(p in text for p in ["delete all files", "give me the code to", "urgent cash", "need cash"]):
        return "invalid"
    if _count_signals(text, FEATURE_REQUEST_SIGNALS) > 0:
        return "feature_request"
    if _count_signals(text, BUG_SIGNALS) > 0:
        return "bug"
    return "product_issue"


def assess_risk(text: str) -> tuple[bool, list[str]]:
    matched = [s for s in RISK_SIGNALS if s in text]
    return bool(matched), matched


def classify(issue: str, subject: str, company_field: str) -> ClassificationResult:
    combined = _normalize(f"{issue} {subject}")

    company = company_field if company_field in COMPANIES else infer_company(combined)

    product_area  = classify_product_area(combined, company)
    request_type  = classify_request_type(combined)
    is_high_risk, risk_reasons = assess_risk(combined)

    return ClassificationResult(
        company=company,
        product_area=product_area,
        request_type=request_type,
        is_high_risk=is_high_risk,
        risk_reasons=risk_reasons,
    )
