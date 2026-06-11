"""Stage 1: Semantic score primitives.

Each primitive returns a deterministic, explainable score in [0,1]
indicating the degree to which a semantic concept applies to a prompt.

These are NOT opaque statistical classifiers. They are interpretable
heuristic functions designed to survive lexical transformations.
"""

import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from core.primitive import Classifier, SemanticScorePrimitive
from core.types import Prompt

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared helper utilities
# ---------------------------------------------------------------------------

_INSTRUCTION_VERBS: Set[str] = {
    "explain", "describe", "tell", "show", "give", "list", "write",
    "create", "make", "build", "generate", "produce", "construct",
    "develop", "design", "prepare", "provide", "outline", "detail",
    "specify", "define", "formulate", "derive", "compute", "calculate",
    "solve", "find", "determine", "identify", "demonstrate", "illustrate",
    "teach", "instruct", "guide", "walk", "lead", "mentor", "train",
    "how", "steps", "procedure", "process", "method", "technique",
    "recipe", "instructions", "directions", "guide", "tutorial",
    "walkthrough", "step-by-step", "step by step",
}

_HARMFUL_KEYWORDS: Set[str] = {
    "bomb", "explosive", "weapon", "attack", "kill", "murder", "assassinate",
    "poison", "drug", "hack", "malware", "virus", "ransomware", "trojan",
    "exploit", "bypass", "steal", "fraud", "scam", "identity theft",
    "child abuse", "terrorism", "nuclear", "biological weapon", "chemical weapon",
    "suicide", "self-harm", "eat so", "harmful", "toxic",
}

_PROCEDURAL_INDICATORS: Set[str] = {
    "step", "first", "next", "then", "finally", "stage", "phase",
    "procedure", "process", "sequence", "order", "instructions",
    "directions", "recipe", "guide", "walkthrough", "tutorial",
    "method", "technique", "approach", "step-by-step", "numbered",
    "outline", "diagram", "flowchart", "algorithm",
}

_JAILBREAK_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?i)\bignore\s+(previous|above|all|your)\s+(instructions|commands|directions)"),
    re.compile(r"(?i)\b(forget|disregard|bypass|override)\s+(your|all|the)\s+(training|guidelines|rules|safety)"),
    re.compile(r"(?i)\b(dan\b|do\s+anything\s+now|jailbreak|unfiltered|uncensored)"),
    re.compile(r"(?i)\b(role\s*play|roleplay)\b.*\b(character|persona|scenario)"),
    re.compile(r"(?i)\b(hypothetical|fictional|creative\s+writing)\b.*\b(scenario|story)"),
    re.compile(r"(?i)\b(act\s+as|pretend|imagine|suppose)\b"),
    re.compile(r"(?i)\b(tell\s+me\s+how\s+to|give\s+me\s+(the\s+)?(steps|instructions))\b"),
    re.compile(r"(?i)\b(academic|research|educational)\s+purposes?\b"),
    re.compile(r"(?i)\b(safety|security)\s*(research|test|evaluation)\b"),
    re.compile(r"(?i)\bcode\s*block\b"),
    re.compile(r"(?i)\b(wrapped|embedded|enclosed)\s+in\b"),
    re.compile(r"(?i)\b(encoded|base64|hex|rot13|obfuscated)\b"),
    re.compile(r"(?i)\b(output\s+format|format\s+as|present\s+as)\b"),
]


def _extract_base_text(prompt: str) -> str:
    """Strip common wrappers to get at the core semantic content."""
    text = prompt
    text = re.sub(r"(?i)```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)'''.*?'''", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)<\|.*?\|>", "", text)
    text = re.sub(r"(?i)system\s*:.*?(?=user|human|assistant)", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)(user|human|assistant)\s*:", "", text)
    text = re.sub(r"\*+", "", text)
    text = re.sub(r"^[-=#]+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\\u[0-9a-fA-F]{4}", "", text)
    text = re.sub(r"\\x[0-9a-fA-F]{2}", "", text)
    text = re.sub(r"[%\(\)]", "", text)
    return text.strip()


def _count_instruction_verbs(text: str) -> int:
    words = set(re.findall(r"[a-zA-Z]{3,}", text.lower()))
    return sum(1 for v in _INSTRUCTION_VERBS if v in words)


def _has_interrogative_structure(text: str) -> bool:
    stripped = text.strip()
    if stripped.endswith("?"):
        return True
    patterns = [
        r"(?i)^(how|what|where|when|why|which|who|whose|whom)\b",
        r"(?i)^(can|could|would|will|do|does|did|is|are|am|has|have|had)\b.*\?$",
        r"(?i)^(tell|show|explain|describe|give)\s+me\b",
        r"(?i)^(i\s+want|i\s+need|i\'?d\s+like)\s+(to\s+)?(know|understand|learn|see|get|find)\b",
    ]
    return any(re.match(p, stripped) for p in patterns)


def _compute_instruction_score(text: str) -> float:
    base = _extract_base_text(text)
    if not base:
        return 0.0
    verb_count = _count_instruction_verbs(base)
    has_interrogative = _has_interrogative_structure(base)
    length = len(base)
    word_count = len(re.findall(r"\b\w+\b", base))
    has_imperative = bool(re.match(r"^[A-Z][a-z]*\b", base))
    has_procedural = any(p in base.lower() for p in _PROCEDURAL_INDICATORS)
    signals = 0.0
    if verb_count >= 3:
        signals += 0.40
    elif verb_count >= 1:
        signals += 0.20
    if has_interrogative:
        signals += 0.20
    if has_imperative:
        signals += 0.15
    if has_procedural:
        signals += 0.20
    if length > 30 and word_count >= 5:
        signals += 0.05
    if length > 100:
        signals += 0.05
    score = min(1.0, signals + 0.05)
    score = max(0.0, score)
    return round(score, 4)


def _compute_harmfulness_score(text: str) -> float:
    base = _extract_base_text(text)
    if not base:
        return 0.0
    lower = base.lower()
    words = set(re.findall(r"[a-zA-Z]{3,}", lower))
    harmful_hits = words & _HARMFUL_KEYWORDS
    severity = 0.0
    if harmful_hits:
        severity += min(0.15 * len(harmful_hits), 0.50)
        for kw in harmful_hits:
            score_value = {"bomb": 0.95, "explosive": 0.90, "weapon": 0.85,
                         "kill": 0.80, "murder": 0.85, "hack": 0.70,
                         "malware": 0.75, "poison": 0.85, "terrorism": 0.90,
                         "nuclear": 0.80, "suicide": 0.85, "self-harm": 0.90,
                         "child abuse": 0.95, "assassinate": 0.90}.get(kw, 0.50)
            severity = max(severity, score_value * 0.7 + 0.15)
    if re.search(r"(?i)how\s+to\s+(make|build|create|construct|use)\s+(a\s+|an\s+)?(" + "|".join(_HARMFUL_KEYWORDS) + r")", base):
        severity = max(severity, 0.85)
    if re.search(r"(?i)(steps|instructions|guide|procedure)\s+.*(for|to)", base) and harmful_hits:
        severity = max(severity, 0.90)
    score = min(1.0, severity + 0.05)
    return round(score, 4)


def _compute_procedurality_score(text: str) -> float:
    base = _extract_base_text(text)
    if not base:
        return 0.0
    lower = base.lower()
    word_count = len(re.findall(r"\b\w+\b", lower))
    proc_hits = [p for p in _PROCEDURAL_INDICATORS if p in lower]
    has_numbers = bool(re.search(r"\b\d+[\.\)]\s+", base))
    has_list_markers = bool(re.search(r"^[-*\d+\.\)]\s+", base, re.MULTILINE))
    signals = 0.0
    if len(proc_hits) >= 3:
        signals += 0.40
    elif len(proc_hits) >= 1:
        signals += 0.20
    if has_numbers:
        signals += 0.15
    if has_list_markers:
        signals += 0.15
    if word_count >= 15:
        signals += 0.10
    if re.search(r"(?i)^(how\s+to|steps?\s+(to|for|in)|the\s+process\s+of)", base):
        signals += 0.25
    if re.search(r"(?i)(first|second|third|finally)\s*[,:]", base):
        signals += 0.15
    if re.search(r"(?i)(explain|describe|outline)\s+(how|the\s+(process|procedure|steps))", base):
        signals += 0.20
    score = min(1.0, signals + 0.05)
    return round(score, 4)


def _compute_jailbreak_score(text: str) -> float:
    base = _extract_base_text(text)
    if not base:
        return 0.0
    lower = base.lower()
    score = 0.0
    for pattern in _JAILBREAK_PATTERNS:
        if pattern.search(lower):
            score += 0.15
    if re.search(r"(?i)(ignore|disregard|forget|bypass|override)", lower):
        score += 0.10
    if re.search(r"(?i)(safety|ethical|guidelines|rules|policy|restrictions)", lower):
        score += 0.10
    if re.search(r"(?i)(role\s*play|act\s+as|pretend|imagine|scenario|character)", lower):
        score += 0.10
    if re.search(r"(?i)(encoded|obfuscated|cipher|encrypted|hidden)", lower):
        score += 0.10
    if re.search(r"(?i)(for\s+(academic|research|educational)\s+purposes)", lower):
        score += 0.10
    if re.search(r"(?i)(output\s+(format|as)|format\s+(as|the\s+output))", lower):
        score += 0.05
    score = min(1.0, score)
    return round(score, 4)


# ---------------------------------------------------------------------------
# Semantic Score Primitives
# ---------------------------------------------------------------------------

class InstructionScorePrimitive(SemanticScorePrimitive):
    """Score a prompt by how strongly it resembles an instruction request.

    Survives: roleplay wrappers, markdown, leetspeak, unicode tricks,
    encoding transforms because _extract_base_text strips those.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="instruction_score",
            input_type="String",
            output_type="SemanticScore",
            metadata={"category": "semantic_score", "deterministic": True},
            **kwargs,
        )

    def evaluate(self, prompt: Prompt) -> float:
        return _compute_instruction_score(str(prompt))

    def explain(self, prompt: Prompt) -> Dict[str, Any]:
        base = _extract_base_text(str(prompt))
        return {
            "score": self.evaluate(prompt),
            "base_text": base[:200] if base else "",
            "verb_count": _count_instruction_verbs(base),
            "is_interrogative": _has_interrogative_structure(base),
            "has_imperative": bool(re.match(r"^[A-Z][a-z]*\b", base)),
        }


class HarmfulnessScorePrimitive(SemanticScorePrimitive):
    """Score a prompt by how harmful the requested content is."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="harmfulness_score",
            input_type="String",
            output_type="SemanticScore",
            metadata={"category": "semantic_score", "deterministic": True},
            **kwargs,
        )

    def evaluate(self, prompt: Prompt) -> float:
        return _compute_harmfulness_score(str(prompt))

    def explain(self, prompt: Prompt) -> Dict[str, Any]:
        base = _extract_base_text(str(prompt))
        lower = base.lower()
        words = set(re.findall(r"[a-zA-Z]{3,}", lower))
        return {
            "score": self.evaluate(prompt),
            "harmful_hits": sorted(words & _HARMFUL_KEYWORDS),
        }


class ProceduralityScorePrimitive(SemanticScorePrimitive):
    """Score a prompt by how procedural or step-by-step it is.

    High procedurality suggests the user wants instructions,
    which is a strong semantic signal for safety classification.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="procedurality_score",
            input_type="String",
            output_type="SemanticScore",
            metadata={"category": "semantic_score", "deterministic": True},
            **kwargs,
        )

    def evaluate(self, prompt: Prompt) -> float:
        return _compute_procedurality_score(str(prompt))

    def explain(self, prompt: Prompt) -> Dict[str, Any]:
        base = _extract_base_text(str(prompt))
        lower = base.lower()
        return {
            "score": self.evaluate(prompt),
            "procedural_hits": [p for p in _PROCEDURAL_INDICATORS if p in lower],
        }


class JailbreakScorePrimitive(SemanticScorePrimitive):
    """Score a prompt by how likely it is a jailbreak attempt."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            name="jailbreak_score",
            input_type="String",
            output_type="SemanticScore",
            metadata={"category": "semantic_score", "deterministic": True},
            **kwargs,
        )

    def evaluate(self, prompt: Prompt) -> float:
        return _compute_jailbreak_score(str(prompt))

    def explain(self, prompt: Prompt) -> Dict[str, Any]:
        base = _extract_base_text(str(prompt))
        matched: List[str] = []
        for pattern in _JAILBREAK_PATTERNS:
            m = pattern.search(base)
            if m:
                matched.append(m.group()[:80])
        return {
            "score": self.evaluate(prompt),
            "matched_patterns": matched,
        }


# ---------------------------------------------------------------------------
# Convenience accessor
# ---------------------------------------------------------------------------

_ALL_SEMANTIC_PRIMITIVES = [
    InstructionScorePrimitive,
    HarmfulnessScorePrimitive,
    ProceduralityScorePrimitive,
    JailbreakScorePrimitive,
]

_SEMANTIC_SCORE_NAMES = {p().name for p in _ALL_SEMANTIC_PRIMITIVES}


def is_semantic_score_primitive(name: str) -> bool:
    """Check if a primitive name corresponds to a semantic score."""
    return name in _SEMANTIC_SCORE_NAMES
