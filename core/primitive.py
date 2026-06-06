"""Primitive types for HARMONY-X: Predicate, Transform, Classifier with 92 built-in primitives."""

import abc
import base64
import html
import json
import logging
import math
import random
import re
import statistics
import string
import time
import urllib.parse
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Set, Tuple, Type

from .types import Prompt

logger = logging.getLogger(__name__)

PrimitiveParameters = Dict[str, Any]
PrimitiveType = Literal["Boolean", "Numeric", "String", "TransformResult", "ClassifierScore"]


class Primitive(abc.ABC):
    name: str
    parameters: PrimitiveParameters
    input_type: PrimitiveType
    output_type: PrimitiveType
    metadata: Dict[str, Any]
    version_id: str
    created_at: float
    deprecated_at: Optional[float]

    def __init__(
        self,
        name: str,
        parameters: Optional[PrimitiveParameters] = None,
        input_type: PrimitiveType = "String",
        output_type: PrimitiveType = "String",
        metadata: Optional[Dict[str, Any]] = None,
        version_id: str = "1.0",
        created_at: Optional[float] = None,
        deprecated_at: Optional[float] = None,
    ) -> None:
        self.name = name
        self.parameters = parameters or {}
        self.input_type = input_type
        self.output_type = output_type
        self.metadata = metadata or {}
        self.version_id = version_id
        self.created_at = created_at or time.time()
        self.deprecated_at = deprecated_at

    @abc.abstractmethod
    def evaluate(self, prompt: Prompt) -> Any:
        raise NotImplementedError

    @property
    def type_signature(self) -> str:
        return f"{self.input_type} -> {self.output_type}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "parameters": self.parameters,
            "type": type(self).__name__,
            "input_type": self.input_type,
            "output_type": self.output_type,
            "version_id": getattr(self, "version_id", "1.0"),
            "created_at": getattr(self, "created_at", 0.0),
            "deprecated_at": getattr(self, "deprecated_at", None),
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, parameters={self.parameters!r}, "
            f"output_type={self.output_type!r})"
        )


class Predicate(Primitive):
    def evaluate(self, prompt: Prompt) -> bool:
        return False

class Transform(Primitive):
    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt

class Classifier(Primitive):
    def evaluate(self, prompt: Prompt) -> float:
        return 0.0


class PrimitiveRegistry:
    _instance: ClassVar[Optional["PrimitiveRegistry"]] = None

    def __new__(cls) -> "PrimitiveRegistry":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._registry = {}
        return cls._instance

    def register(self, primitive_cls: Type[Primitive]) -> None:
        self._registry[primitive_cls.__name__] = primitive_cls
        try:
            sample = primitive_cls()
            self._registry[sample.name] = primitive_cls
        except Exception:
            pass

    def get(self, name: str, parameters: Optional[PrimitiveParameters] = None) -> Primitive:
        primitive_cls = self._registry.get(name)
        if primitive_cls is None:
            for candidate in self._registry.values():
                try:
                    instance = candidate()
                    if getattr(instance, "name", None) == name:
                        primitive_cls = candidate
                        break
                except Exception:
                    continue
        if primitive_cls is None:
            raise ValueError(f"Unknown primitive '{name}'")
        try:
            return primitive_cls(**(parameters or {}))
        except TypeError:
            return primitive_cls()

    def list_primitives(self) -> List[str]:
        primitive_names = set()
        for primitive_cls in self._registry.values():
            try:
                primitive_names.add(primitive_cls().name)
            except Exception:
                primitive_names.add(primitive_cls.__name__)
        return sorted(primitive_names)


# =============================================================================
# PREDICATES (27)
# =============================================================================

@dataclass
class ContainsWordPredicate(Predicate):
    word: str = field(default="")
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        self.name = "contains_word"
        self.parameters = {"word": self.word, "case_sensitive": self.case_sensitive}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether a prompt includes a specific word.",
            "category": "predicate",
            "related_primitives": ["contains_any_word", "matches_regex"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        if self.case_sensitive:
            return self.word in prompt
        return self.word.lower() in prompt.lower()


@dataclass
class ContainsAnyWordPredicate(Predicate):
    words: List[str] = field(default_factory=list)
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        self.name = "contains_any_word"
        self.parameters = {"words": self.words, "case_sensitive": self.case_sensitive}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether a prompt contains at least one word from a list.",
            "category": "predicate",
            "related_primitives": ["contains_word", "contains_all_words"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        text = prompt if self.case_sensitive else prompt.lower()
        words = self.words if self.case_sensitive else [w.lower() for w in self.words]
        return any(w in text for w in words)


@dataclass
class ContainsAllWordsPredicate(Predicate):
    words: List[str] = field(default_factory=list)
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        self.name = "contains_all_words"
        self.parameters = {"words": self.words, "case_sensitive": self.case_sensitive}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether a prompt contains every word from a list.",
            "category": "predicate",
            "related_primitives": ["contains_word", "contains_any_word"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        text = prompt if self.case_sensitive else prompt.lower()
        words = self.words if self.case_sensitive else [w.lower() for w in self.words]
        return all(w in text for w in words)


@dataclass
class LengthGtPredicate(Predicate):
    threshold: int = field(default=0)

    def __post_init__(self) -> None:
        self.name = "length_gt"
        self.parameters = {"threshold": self.threshold}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt length exceeds a threshold.",
            "category": "predicate",
            "related_primitives": ["length_lt"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return len(prompt) > self.threshold


@dataclass
class LengthLtPredicate(Predicate):
    threshold: int = field(default=0)

    def __post_init__(self) -> None:
        self.name = "length_lt"
        self.parameters = {"threshold": self.threshold}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt length is below a threshold.",
            "category": "predicate",
            "related_primitives": ["length_gt"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return len(prompt) < self.threshold


@dataclass
class MatchesRegexPredicate(Predicate):
    pattern: str = field(default="")

    def __post_init__(self) -> None:
        self.name = "matches_regex"
        self.parameters = {"pattern": self.pattern}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt matches a regular expression.",
            "category": "predicate",
            "related_primitives": ["contains_word"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        try:
            return re.search(self.pattern, prompt) is not None
        except re.error:
            return False


@dataclass
class StartsWithPredicate(Predicate):
    prefix: str = field(default="")
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        self.name = "starts_with"
        self.parameters = {"prefix": self.prefix, "case_sensitive": self.case_sensitive}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt starts with a specific prefix.",
            "category": "predicate",
            "related_primitives": ["ends_with"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        if self.case_sensitive:
            return prompt.startswith(self.prefix)
        return prompt.lower().startswith(self.prefix.lower())


@dataclass
class EndsWithPredicate(Predicate):
    suffix: str = field(default="")
    case_sensitive: bool = False

    def __post_init__(self) -> None:
        self.name = "ends_with"
        self.parameters = {"suffix": self.suffix, "case_sensitive": self.case_sensitive}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt ends with a specific suffix.",
            "category": "predicate",
            "related_primitives": ["starts_with"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        if self.case_sensitive:
            return prompt.endswith(self.suffix)
        return prompt.lower().endswith(self.suffix.lower())


@dataclass
class HasNumberPredicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "has_number"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt contains any digit (0-9).",
            "category": "predicate",
            "related_primitives": ["has_special_char"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return any(c.isdigit() for c in prompt)


@dataclass
class HasSpecialCharPredicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "has_special_char"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt contains non-alphanumeric, non-space characters.",
            "category": "predicate",
            "related_primitives": ["has_number"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return any(not c.isalnum() and not c.isspace() for c in prompt)


@dataclass
class IsAllCapsPredicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "is_all_caps"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt is entirely uppercase (ignoring non-letters).",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        letters = [c for c in prompt if c.isalpha()]
        return len(letters) > 0 and all(c.isupper() for c in letters)


@dataclass
class ContainsLeetPredicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "contains_leet"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Detect leetspeak: digits/ symbols substituted for letters (e→3, a→4, etc.).",
            "category": "predicate",
            "related_primitives": [],
        }

    _LEET_DIGITS: ClassVar[Set[str]] = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}
    _LEET_SYMBOLS: ClassVar[Set[str]] = {"@", "$", "!", "+", "|"}

    def evaluate(self, prompt: Prompt) -> bool:
        leet_count = 0
        for c in prompt:
            if c in self._LEET_DIGITS or c in self._LEET_SYMBOLS:
                leet_count += 1
        return len(prompt) > 3 and leet_count / max(len(prompt), 1) > 0.1


@dataclass
class ContainsRot13Predicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "contains_rot13"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Detect if text appears ROT13-encoded (only a-zA-Z, shifted pattern).",
            "category": "predicate",
            "related_primitives": ["contains_base64", "contains_hex"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        letters = [c for c in prompt if c.isalpha()]
        if len(letters) < 5:
            return False
        rot_count = sum(1 for c in letters if c.lower() in "nopqrstuvwxyz")
        return rot_count / len(letters) > 0.5


@dataclass
class ContainsBase64Predicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "contains_base64"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Detect if text appears Base64-encoded (length >4, valid chars, = padding).",
            "category": "predicate",
            "related_primitives": ["contains_rot13", "contains_hex"],
        }

    _B64: ClassVar[Set[str]] = set(string.ascii_letters + string.digits + "+/=")

    def evaluate(self, prompt: Prompt) -> bool:
        stripped = prompt.strip()
        if len(stripped) < 5:
            return False
        valid = sum(1 for c in stripped if c in self._B64)
        ratio = valid / len(stripped)
        return ratio > 0.95


@dataclass
class ContainsHexPredicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "contains_hex"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Detect if text appears hex-encoded (only 0-9 A-F a-f, even length tokens).",
            "category": "predicate",
            "related_primitives": ["contains_base64", "contains_rot13"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        tokens = re.findall(r'[0-9A-Fa-f]{6,}', prompt)
        if not tokens:
            return False
        hex_chars = sum(1 for c in prompt if c in string.hexdigits)
        return hex_chars / max(len(prompt), 1) > 0.8 and len(tokens) >= 1


@dataclass
class IsEmptyPredicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "is_empty"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt is empty or only whitespace.",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return len(prompt.strip()) == 0


@dataclass
class StartsWithRoleplayPredicate(Predicate):
    _ROLEPLAY_PREFIXES: ClassVar[List[str]] = [
        "as a", "pretend", "imagine you are", "act as", "you are now",
        "from now on", "roleplay", "let's roleplay", "scenario:",
        "you will act", "you are going to",
    ]

    def __post_init__(self) -> None:
        self.name = "starts_with_roleplay"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt starts with a role-play pattern (e.g. 'As a...', 'Pretend...').",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        lower = prompt.lower().strip()
        return any(lower.startswith(p) for p in self._ROLEPLAY_PREFIXES)


@dataclass
class ContainsSystemOverridePredicate(Predicate):
    _OVERRIDE_PHRASES: ClassVar[List[str]] = [
        "ignore previous", "ignore all", "disregard", "forget",
        "override", "new instruction", "system prompt",
        "you are not", "ignore the above", "don't follow",
        "break character", "do not follow",
    ]

    def __post_init__(self) -> None:
        self.name = "contains_system_override"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Detect instructions attempting to override system prompts.",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        lower = prompt.lower()
        return any(phrase in lower for phrase in self._OVERRIDE_PHRASES)


@dataclass
class ContainsDelimiterPredicate(Predicate):
    delimiters: List[str] = field(default_factory=lambda: ['"""', "---", "|||", "===", "```"])

    def __post_init__(self) -> None:
        self.name = "contains_delimiter"
        self.parameters = {"delimiters": self.delimiters}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt contains injection delimiters like triple quotes or separators.",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return any(d in prompt for d in self.delimiters)


@dataclass
class ContainsCodeBlockPredicate(Predicate):
    def __post_init__(self) -> None:
        self.name = "contains_code_block"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt contains markdown code blocks (```).",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return "```" in prompt


@dataclass
class HasEmojiPredicate(Predicate):
    _EMOJI_PATTERN: ClassVar[re.Pattern] = re.compile(
        "[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF\U0000200D\uFE0F]"
    )

    def __post_init__(self) -> None:
        self.name = "has_emoji"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt contains emoji characters.",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return bool(self._EMOJI_PATTERN.search(prompt))


@dataclass
class ContainsURLPredicate(Predicate):
    _URL_PATTERN: ClassVar[re.Pattern] = re.compile(
        r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:/\S*)?'
    )

    def __post_init__(self) -> None:
        self.name = "contains_url"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt contains a URL or web address.",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return bool(self._URL_PATTERN.search(prompt))


@dataclass
class SentimentPredicate(Predicate):
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.name = "sentiment"
        self.parameters = {"threshold": self.threshold}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Compare sentiment score against a threshold (>= threshold = True).",
            "category": "predicate",
            "related_primitives": ["sentiment_score"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return _estimate_sentiment(prompt) >= self.threshold


@dataclass
class IntentPredicate(Predicate):
    intent_type: str = "harmful"

    def __post_init__(self) -> None:
        self.name = "intent"
        self.parameters = {"intent_type": self.intent_type}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt intent matches a type (harmful, innocuous, etc.).",
            "category": "predicate",
            "related_primitives": ["intent_score"],
        }

    _HARMFUL_KEYWORDS: ClassVar[List[str]] = [
        "bomb", "kill", "weapon", "hack", "steal", "virus", "drug",
        "murder", "exploit", "attack", "jailbreak", "bypass",
    ]

    def evaluate(self, prompt: Prompt) -> bool:
        if self.intent_type == "harmful":
            return any(kw in prompt.lower() for kw in self._HARMFUL_KEYWORDS)
        return True


@dataclass
class MatchesJailbreakPatternPredicate(Predicate):
    pattern_library: str = "harmbench"

    def __post_init__(self) -> None:
        self.name = "matches_jailbreak_pattern"
        self.parameters = {"pattern_library": self.pattern_library}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether prompt matches known jailbreak patterns from a library.",
            "category": "predicate",
            "related_primitives": [],
        }

    _PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(r"ignore\s+(all\s+)?(previous|above|prior)", re.I),
        re.compile(r"dan|do\s+anything\s+now", re.I),
        re.compile(r"you\s+(must|have\s+to|will)\s+obey", re.I),
        re.compile(r"character\s+mode|hypothetical\s+scenario", re.I),
        re.compile(r"system\s+prompt\s*:", re.I),
        re.compile(r"new\s+instruction", re.I),
        re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
    ]

    def evaluate(self, prompt: Prompt) -> bool:
        return any(p.search(prompt) for p in self._PATTERNS)


@dataclass
class ContainsEncodingWrapperPredicate(Predicate):
    _WRAPPER_PATTERN: ClassVar[re.Pattern] = re.compile(
        r"(```(base64|hex|rot13|binary|morse)\s*\n.*?\n```)", re.DOTALL | re.IGNORECASE
    )

    def __post_init__(self) -> None:
        self.name = "contains_encoding_wrapper"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Detect if prompt is wrapped in an encoding block like ```base64...```.",
            "category": "predicate",
            "related_primitives": ["contains_code_block"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return bool(self._WRAPPER_PATTERN.search(prompt))


@dataclass
class IsRepetitivePredicate(Predicate):
    threshold: int = 10

    def __post_init__(self) -> None:
        self.name = "is_repetitive"
        self.parameters = {"threshold": self.threshold}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.metadata = {
            "description": "Check whether any n-gram repeats more than threshold times.",
            "category": "predicate",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        tokens = prompt.split()
        if len(tokens) < 3:
            return False
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        counts = Counter(bigrams)
        return max(counts.values()) > self.threshold


# =============================================================================
# TRANSFORMS (38)
# =============================================================================

@dataclass
class Rot13Transform(Transform):
    def __post_init__(self) -> None:
        self.name = "rot13"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Apply ROT13 cipher to the prompt.",
            "category": "transform",
            "related_primitives": ["caesar_cipher", "atbash_cipher"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.translate(
            str.maketrans(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
            )
        )


@dataclass
class Base64EncodeTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "base64"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Encode prompt to Base64.",
            "category": "transform",
            "related_primitives": ["base64_decode", "hex_encode"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return base64.b64encode(prompt.encode("utf-8")).decode("ascii")


@dataclass
class Base64DecodeTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "base64_decode"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Decode a prompt from Base64.",
            "category": "transform",
            "related_primitives": ["base64", "hex_encode"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        try:
            decoded_bytes = base64.b64decode(prompt, validate=True)
            return decoded_bytes.decode("utf-8", errors="ignore")
        except Exception:
            return prompt


@dataclass
class ToLowercaseTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "to_lowercase"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Lowercase the prompt text.",
            "category": "transform",
            "related_primitives": ["to_uppercase"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.lower()


@dataclass
class ToUppercaseTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "to_uppercase"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Uppercase the prompt text.",
            "category": "transform",
            "related_primitives": ["to_lowercase"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.upper()


@dataclass
class RemovePunctuationTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "remove_punctuation"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Remove punctuation from the prompt.",
            "category": "transform",
            "related_primitives": ["remove_whitespace"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.translate(str.maketrans("", "", string.punctuation))


@dataclass
class LeetSpeakTransform(Transform):
    level: int = 1

    def __post_init__(self) -> None:
        self.name = "leet_speak"
        self.parameters = {"level": self.level}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Convert text to leetspeak (e→3, a→4, etc.).",
            "category": "transform",
            "related_primitives": [],
        }

    _BASIC: ClassVar[Dict[str, str]] = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7"}
    _HEAVY: ClassVar[Dict[str, str]] = {
        "a": "@", "b": "8", "e": "3", "g": "9", "i": "1", "l": "1",
        "o": "0", "s": "$", "t": "7", "z": "2",
    }

    def evaluate(self, prompt: Prompt) -> Prompt:
        mapping = self._HEAVY if self.level >= 2 else self._BASIC
        result = []
        for c in prompt:
            result.append(mapping.get(c.lower(), c))
        return "".join(result)


@dataclass
class ReverseTextTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "reverse_text"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Reverse the entire string.",
            "category": "transform",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt[::-1]


@dataclass
class PigLatinTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "pig_latin"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Convert text to Pig Latin.",
            "category": "transform",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        vowels = "aeiouAEIOU"
        words = prompt.split()
        result = []
        for word in words:
            if not word:
                continue
            if word[0] in vowels:
                result.append(word + "way")
            else:
                first_vowel = next((i for i, c in enumerate(word) if c in vowels), len(word))
                result.append(word[first_vowel:] + word[:first_vowel] + "ay")
        return " ".join(result)


@dataclass
class MorseCodeTransform(Transform):
    encode: bool = True

    def __post_init__(self) -> None:
        self.name = "morse_code"
        self.parameters = {"encode": self.encode}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Convert text to/from Morse code.",
            "category": "transform",
            "related_primitives": [],
        }

    _TO_MORSE: ClassVar[Dict[str, str]] = {
        "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
        "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
        "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
        "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
        "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
        "Z": "--..", "0": "-----", "1": ".----", "2": "..---", "3": "...--",
        "4": "....-", "5": ".....", "6": "-....", "7": "--...", "8": "---..",
        "9": "----.", ".": ".-.-.-", ",": "--..--", "?": "..--..",
        "!": "-.-.--", ":": "---...", ";": "-.-.-.", "'": ".----.",
    }
    _FROM_MORSE: ClassVar[Dict[str, str]] = {v: k for k, v in _TO_MORSE.items()}

    def evaluate(self, prompt: Prompt) -> Prompt:
        if self.encode:
            words = prompt.upper().split()
            encoded = []
            for word in words:
                chars = [self._TO_MORSE.get(c, c) for c in word]
                encoded.append(" ".join(chars))
            return " / ".join(encoded)
        else:
            words = prompt.split(" / ")
            decoded = []
            for word in words:
                chars = word.split()
                decoded.append("".join(self._FROM_MORSE.get(c, c) for c in chars))
            return " ".join(decoded)


@dataclass
class AddPrefixTransform(Transform):
    prefix: str = ""

    def __post_init__(self) -> None:
        self.name = "add_prefix"
        self.parameters = {"prefix": self.prefix}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Prepend text to the prompt.",
            "category": "transform",
            "related_primitives": ["add_suffix", "add_role_play"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return self.prefix + prompt


@dataclass
class AddSuffixTransform(Transform):
    suffix: str = ""

    def __post_init__(self) -> None:
        self.name = "add_suffix"
        self.parameters = {"suffix": self.suffix}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Append text to the prompt.",
            "category": "transform",
            "related_primitives": ["add_prefix"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt + self.suffix


@dataclass
class WrapCodeBlockTransform(Transform):
    language: str = "python"

    def __post_init__(self) -> None:
        self.name = "wrap_code_block"
        self.parameters = {"language": self.language}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Wrap prompt in a markdown code block with language hint.",
            "category": "transform",
            "related_primitives": ["format_as_json"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return f"```{self.language}\n{prompt}\n```"


@dataclass
class InsertTyposTransform(Transform):
    probability: float = 0.1

    def __post_init__(self) -> None:
        self.name = "insert_typos"
        self.parameters = {"probability": self.probability}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Randomly insert typos (swap/delete adjacent chars) at given probability.",
            "category": "transform",
            "related_primitives": ["insert_synonyms"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        chars = list(prompt)
        for i in range(len(chars) - 1):
            if random.random() < self.probability:
                chars[i], chars[i+1] = chars[i+1], chars[i]
        return "".join(chars)


@dataclass
class WordShuffleTransform(Transform):
    seed: int = 42

    def __post_init__(self) -> None:
        self.name = "word_shuffle"
        self.parameters = {"seed": self.seed}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Shuffle word order deterministically by seed.",
            "category": "transform",
            "related_primitives": ["reverse_text"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        words = prompt.split()
        rng = random.Random(self.seed)
        rng.shuffle(words)
        return " ".join(words)


@dataclass
class AddMarkdownTransform(Transform):
    style: str = "bold"

    def __post_init__(self) -> None:
        self.name = "add_markdown"
        self.parameters = {"style": self.style}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Add markdown formatting (bold, italic, code) around the prompt.",
            "category": "transform",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        if self.style == "bold":
            return f"**{prompt}**"
        if self.style == "italic":
            return f"*{prompt}*"
        if self.style == "code":
            return f"`{prompt}`"
        return prompt


@dataclass
class AddZeroWidthCharsTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "add_zero_width_chars"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Insert zero-width space (ZWSP) between each character.",
            "category": "transform",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        zwsp = "\u200B"
        return zwsp.join(prompt)


@dataclass
class UnicodeObfuscateTransform(Transform):
    alphabet: str = "cyrillic"

    def __post_init__(self) -> None:
        self.name = "unicode_obfuscate"
        self.parameters = {"alphabet": self.alphabet}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Replace Latin letters with visually similar Cyrillic/Greek homoglyphs.",
            "category": "transform",
            "related_primitives": ["leet_speak"],
        }

    _CYRILLIC_MAP: ClassVar[Dict[str, str]] = {
        "a": "а", "A": "А", "e": "е", "E": "Е", "o": "о", "O": "О",
        "p": "р", "P": "Р", "c": "с", "C": "С", "y": "у", "Y": "У",
        "x": "х", "X": "Х", "i": "і", "I": "І", "k": "к", "K": "К",
        "m": "м", "M": "М", "T": "Т", "H": "Н", "B": "В",
    }
    _GREEK_MAP: ClassVar[Dict[str, str]] = {
        "a": "α", "b": "β", "e": "ε", "i": "ι", "k": "κ",
        "n": "η", "o": "ο", "p": "ρ", "t": "τ", "u": "υ", "x": "ξ",
    }

    def evaluate(self, prompt: Prompt) -> Prompt:
        mapping = self._CYRILLIC_MAP if self.alphabet == "cyrillic" else self._GREEK_MAP
        return "".join(mapping.get(c, c) for c in prompt)


@dataclass
class HtmlEncodeTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "html_encode"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Encode special characters as HTML entities (< → &lt;).",
            "category": "transform",
            "related_primitives": ["url_encode", "quoted_printable"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return html.escape(prompt)


@dataclass
class URLEncodeTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "url_encode"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "URL-encode the prompt (space → %20).",
            "category": "transform",
            "related_primitives": ["html_encode", "quoted_printable"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return urllib.parse.quote(prompt)


@dataclass
class QuotedPrintableTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "quoted_printable"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Encode prompt as quoted-printable (=XX format).",
            "category": "transform",
            "related_primitives": ["html_encode", "url_encode"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        result = []
        for c in prompt:
            if c == " ":
                result.append(" ")
            elif c.isalnum() and ord(c) < 128 or c in "-_.!~*'()":
                result.append(c)
            else:
                result.append(f"={ord(c):02X}")
        return "".join(result)


@dataclass
class BinaryEncodeTransform(Transform):
    separator: str = " "

    def __post_init__(self) -> None:
        self.name = "binary_encode"
        self.parameters = {"separator": self.separator}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Encode each character as 8-bit binary.",
            "category": "transform",
            "related_primitives": ["hex_encode"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return self.separator.join(format(ord(c), "08b") for c in prompt)


@dataclass
class HexEncodeTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "hex_encode"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Encode prompt as hex string.",
            "category": "transform",
            "related_primitives": ["binary_encode", "base64"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.encode("utf-8").hex()


@dataclass
class RemoveVowelsTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "remove_vowels"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Remove all vowels (a, e, i, o, u).",
            "category": "transform",
            "related_primitives": ["remove_punctuation", "remove_whitespace"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return re.sub(r"[aeiouAEIOU]", "", prompt)


@dataclass
class BoustrophedonTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "boustrophedon"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Alternate text direction line-by-line (boustrophedon style).",
            "category": "transform",
            "related_primitives": ["reverse_text"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        lines = prompt.split("\n")
        result = []
        for i, line in enumerate(lines):
            result.append(line if i % 2 == 0 else line[::-1])
        return "\n".join(result)


@dataclass
class AtbashCipherTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "atbash_cipher"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Apply Atbash cipher (A↔Z, B↔Y, ...).",
            "category": "transform",
            "related_primitives": ["rot13", "caesar_cipher"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        result = []
        for c in prompt:
            if "A" <= c <= "Z":
                result.append(chr(ord("Z") - (ord(c) - ord("A"))))
            elif "a" <= c <= "z":
                result.append(chr(ord("z") - (ord(c) - ord("a"))))
            else:
                result.append(c)
        return "".join(result)


@dataclass
class CaesarCipherTransform(Transform):
    shift: int = 3

    def __post_init__(self) -> None:
        self.name = "caesar_cipher"
        self.parameters = {"shift": self.shift}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Apply Caesar cipher with a given shift.",
            "category": "transform",
            "related_primitives": ["rot13", "vigenere_cipher"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        result = []
        for c in prompt:
            if "A" <= c <= "Z":
                result.append(chr((ord(c) - ord("A") + self.shift) % 26 + ord("A")))
            elif "a" <= c <= "z":
                result.append(chr((ord(c) - ord("a") + self.shift) % 26 + ord("a")))
            else:
                result.append(c)
        return "".join(result)


@dataclass
class VigenereCipherTransform(Transform):
    key: str = "secret"

    def __post_init__(self) -> None:
        self.name = "vigenere_cipher"
        self.parameters = {"key": self.key}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Apply Vigenère cipher with a repeating key.",
            "category": "transform",
            "related_primitives": ["caesar_cipher"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        result = []
        key_idx = 0
        if not self.key:
            return prompt
        for c in prompt:
            if "A" <= c <= "Z":
                shift = ord(self.key[key_idx % len(self.key)].upper()) - ord("A")
                result.append(chr((ord(c) - ord("A") + shift) % 26 + ord("A")))
                key_idx += 1
            elif "a" <= c <= "z":
                shift = ord(self.key[key_idx % len(self.key)].lower()) - ord("a")
                result.append(chr((ord(c) - ord("a") + shift) % 26 + ord("a")))
                key_idx += 1
            else:
                result.append(c)
        return "".join(result)


@dataclass
class RailFenceCipherTransform(Transform):
    rails: int = 3

    def __post_init__(self) -> None:
        self.name = "rail_fence_cipher"
        self.parameters = {"rails": self.rails}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Apply Rail Fence cipher with N rails.",
            "category": "transform",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        if self.rails <= 1:
            return prompt
        fence: List[List[str]] = [[] for _ in range(self.rails)]
        idx, step = 0, 1
        for c in prompt:
            fence[idx].append(c)
            if idx == 0:
                step = 1
            elif idx == self.rails - 1:
                step = -1
            idx += step
        return "".join("".join(row) for row in fence)


@dataclass
class RemoveWhitespaceTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "remove_whitespace"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Remove all whitespace (spaces, tabs, newlines).",
            "category": "transform",
            "related_primitives": ["remove_punctuation", "remove_vowels"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return "".join(prompt.split())


@dataclass
class InsertSynonymsTransform(Transform):
    probability: float = 0.3

    def __post_init__(self) -> None:
        self.name = "insert_synonyms"
        self.parameters = {"probability": self.probability}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Replace words with synonyms (using simple built-in map).",
            "category": "transform",
            "related_primitives": ["insert_typos"],
        }

    _SYNONYM_MAP: ClassVar[Dict[str, List[str]]] = {
        "make": ["create", "build", "construct", "fabricate"],
        "kill": ["murder", "slay", "assassinate", "eliminate"],
        "hack": ["crack", "breach", "exploit", "infiltrate"],
        "steal": ["take", "pilfer", "swipe", "appropriate"],
        "bomb": ["explosive", "device", "ordnance", "munition"],
        "bad": ["evil", "malicious", "harmful", "dangerous"],
        "good": ["excellent", "positive", "beneficial", "helpful"],
        "big": ["large", "massive", "enormous", "immense"],
        "small": ["tiny", "miniature", "compact", "petite"],
        "get": ["obtain", "acquire", "procure", "secure"],
        "use": ["utilize", "employ", "leverage", "deploy"],
    }

    def evaluate(self, prompt: Prompt) -> Prompt:
        words = prompt.split()
        result = []
        for w in words:
            w_lower = w.lower()
            if w_lower in self._SYNONYM_MAP and random.random() < self.probability:
                replacement = random.choice(self._SYNONYM_MAP[w_lower])
                if w[0].isupper():
                    replacement = replacement.capitalize()
                result.append(replacement)
            else:
                result.append(w)
        return " ".join(result)


@dataclass
class EscapeQuotesTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "escape_quotes"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Add backslash before single quotes.",
            "category": "transform",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.replace("'", "\\'").replace('"', '\\"')


@dataclass
class FormatAsJsonTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "format_as_json"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Wrap prompt in a JSON object.",
            "category": "transform",
            "related_primitives": ["wrap_code_block"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return json.dumps({"prompt": prompt, "instruction": prompt})


@dataclass
class AddRolePlayTransform(Transform):
    role: str = "researcher"

    def __post_init__(self) -> None:
        self.name = "add_role_play"
        self.parameters = {"role": self.role}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Prepend 'As a [role], ' to the prompt.",
            "category": "transform",
            "related_primitives": ["add_prefix"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return f"As a {self.role}, {prompt}"


@dataclass
class TruncateTransform(Transform):
    max_chars: int = 100

    def __post_init__(self) -> None:
        self.name = "truncate"
        self.parameters = {"max_chars": self.max_chars}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Truncate prompt to max_chars characters.",
            "category": "transform",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt[:self.max_chars]


@dataclass
class PadToLengthTransform(Transform):
    length: int = 100
    pad_char: str = " "

    def __post_init__(self) -> None:
        self.name = "pad_to_length"
        self.parameters = {"length": self.length, "pad_char": self.pad_char}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Pad or truncate prompt to exact length with pad_char.",
            "category": "transform",
            "related_primitives": ["truncate"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        if len(prompt) >= self.length:
            return prompt[:self.length]
        return prompt + self.pad_char * (self.length - len(prompt))


@dataclass
class RandomCaseTransform(Transform):
    seed: int = 42
    probability: float = 0.5

    def __post_init__(self) -> None:
        self.name = "random_case"
        self.parameters = {"seed": self.seed, "probability": self.probability}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Randomly toggle character case with given probability and seed.",
            "category": "transform",
            "related_primitives": ["to_lowercase", "to_uppercase"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        rng = random.Random(self.seed)
        result = []
        for c in prompt:
            if rng.random() < self.probability:
                result.append(c.swapcase())
            else:
                result.append(c)
        return "".join(result)


@dataclass
class CharacterSubstitutionTransform(Transform):
    old_char: str = "a"
    new_char: str = "4"

    def __post_init__(self) -> None:
        self.name = "character_substitution"
        self.parameters = {"old_char": self.old_char, "new_char": self.new_char}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.metadata = {
            "description": "Substitute all occurrences of old_char with new_char.",
            "category": "transform",
            "related_primitives": ["leet_speak"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.replace(self.old_char, self.new_char)


# =============================================================================
# CLASSIFIERS (27)
# =============================================================================

try:
    from textblob import TextBlob as _TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False

_TOXIC_KEYWORDS: List[str] = [
    "bomb", "kill", "attack", "weapon", "hack", "steal", "virus",
    "drug", "launder", "murder", "explode", "terrorist",
]


def _estimate_sentiment(prompt: str) -> float:
    neg_words = ["bad", "terrible", "awful", "hate", "horrible", "evil", "worst"]
    pos_words = ["good", "great", "nice", "love", "wonderful", "happy", "excellent"]
    lower = prompt.lower()
    neg_hits = sum(1 for w in neg_words if w in lower)
    pos_hits = sum(1 for w in pos_words if w in lower)
    total = neg_hits + pos_hits
    if total == 0:
        return 0.5
    return pos_hits / total


@dataclass
class ToxicityScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "toxicity_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Estimate a toxicity score (0=safe, 1=toxic) using keyword heuristic.",
            "category": "classifier",
            "related_primitives": ["sentiment_score", "intent_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if _HAS_TEXTBLOB:
            try:
                blob = _TextBlob(prompt)
                keyword_boost = sum(0.15 for kw in _TOXIC_KEYWORDS if kw in prompt.lower())
                return max(0.0, min(1.0, 0.5 - blob.sentiment.polarity * 0.4 + keyword_boost))
            except Exception:
                pass
        prompt_lower = prompt.lower()
        hits = sum(1 for kw in _TOXIC_KEYWORDS if kw in prompt_lower)
        return min(1.0, hits * 0.25)


@dataclass
class SentimentScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "sentiment_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Estimate sentiment polarity (0=negative, 1=positive).",
            "category": "classifier",
            "related_primitives": ["toxicity_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if _HAS_TEXTBLOB:
            try:
                blob = _TextBlob(prompt)
                return max(0.0, min(1.0, (blob.sentiment.polarity + 1.0) / 2.0))
            except Exception:
                pass
        return _estimate_sentiment(prompt)


@dataclass
class IntentScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "intent_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Estimate likelihood of harmful intent (0=benign, 1=harmful).",
            "category": "classifier",
            "related_primitives": ["toxicity_score"],
        }

    _HARMFUL: ClassVar[List[str]] = [
        "bomb", "kill", "weapon", "hack", "steal", "virus", "drug",
        "murder", "exploit", "attack", "jailbreak", "bypass", "explosion",
    ]

    def evaluate(self, prompt: Prompt) -> float:
        lower = prompt.lower()
        hits = sum(1 for kw in self._HARMFUL if kw in lower)
        return min(1.0, hits * 0.2)


@dataclass
class ObscurityScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "obscurity_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Measure obfuscation level from leetspeak, encoding, and special chars.",
            "category": "classifier",
            "related_primitives": ["entropy_score", "language_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        score = 0.0
        leet_count = sum(1 for c in prompt if c in "0123456789@$!+")
        if len(prompt) > 0:
            score += min(0.4, leet_count / len(prompt))
        non_alpha = sum(1 for c in prompt if not c.isalpha() and not c.isspace())
        score += min(0.3, non_alpha / max(len(prompt), 1))
        score += min(0.3, len(re.findall(r'[^\x00-\x7F]', prompt)) * 0.05)
        return min(1.0, score)


@dataclass
class LengthScoreClassifier(Classifier):
    min_len: int = 10
    max_len: int = 1000

    def __post_init__(self) -> None:
        self.name = "length_score"
        self.parameters = {"min_len": self.min_len, "max_len": self.max_len}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Normalized length score (0=very short/long, 1=ideal range).",
            "category": "classifier",
            "related_primitives": ["repetition_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        n = len(prompt)
        if n < self.min_len:
            return n / self.min_len
        if n > self.max_len:
            return max(0.0, 1.0 - (n - self.max_len) / self.max_len)
        return 1.0


@dataclass
class RepetitionScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "repetition_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Repetition score based on n-gram frequency (1=highly repetitive).",
            "category": "classifier",
            "related_primitives": ["unique_token_ratio"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        tokens = prompt.split()
        if len(tokens) < 4:
            return 0.0
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
        counts = Counter(bigrams)
        if not counts:
            return 0.0
        max_count = max(counts.values())
        total = len(counts)
        return min(1.0, (max_count / max(1, total)) * 2.0)


@dataclass
class EntropyScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "entropy_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Character entropy (high → likely encoded/obfuscated).",
            "category": "classifier",
            "related_primitives": ["obscurity_score", "language_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if not prompt:
            return 0.0
        length = len(prompt)
        freq: Dict[str, int] = {}
        for c in prompt:
            freq[c] = freq.get(c, 0) + 1
        entropy = -sum((count / length) * math.log2(count / length) for count in freq.values())
        return min(1.0, entropy / 6.0)


@dataclass
class LanguageScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "language_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Estimate natural language likelihood based on character distribution.",
            "category": "classifier",
            "related_primitives": ["entropy_score"],
        }

    _ENGLISH_FREQ: ClassVar[Dict[str, float]] = {
        "a": 8.2, "b": 1.5, "c": 2.8, "d": 4.3, "e": 12.7, "f": 2.2,
        "g": 2.0, "h": 6.1, "i": 7.0, "j": 0.15, "k": 0.77, "l": 4.0,
        "m": 2.4, "n": 6.7, "o": 7.5, "p": 1.9, "q": 0.095, "r": 6.0,
        "s": 6.3, "t": 9.1, "u": 2.8, "v": 0.98, "w": 2.4, "x": 0.15,
        "y": 2.0, "z": 0.074,
    }

    def evaluate(self, prompt: Prompt) -> float:
        if not prompt:
            return 0.0
        letters = [c.lower() for c in prompt if c.isalpha()]
        if not letters:
            return 0.0
        total = len(letters)
        score = 0.0
        for c, expected in self._ENGLISH_FREQ.items():
            actual = (letters.count(c) / total) * 100
            score += 1.0 - min(1.0, abs(actual - expected) / max(expected, 0.1))
        return max(0.0, min(1.0, score / len(self._ENGLISH_FREQ)))


@dataclass
class JailbreakLikelihoodClassifier(Classifier):
    database: str = "harmbench"

    def __post_init__(self) -> None:
        self.name = "jailbreak_likelihood"
        self.parameters = {"database": self.database}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Score based on known jailbreak pattern density.",
            "category": "classifier",
            "related_primitives": ["prompt_injection_likelihood", "roleplay_likelihood"],
        }

    _PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(r"ignore\s+(all\s+)?(previous|above|prior)", re.I),
        re.compile(r"dan|do\s+anything\s+now", re.I),
        re.compile(r"you\s+(must|have\s+to|will)\s+obey", re.I),
        re.compile(r"character\s+mode|hypothetical\s+scenario", re.I),
        re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
        re.compile(r"system\s+prompt\s*:", re.I),
        re.compile(r"new\s+instruction", re.I),
        re.compile(r"output\s+(without|with\s+no)\s+(censorship|filter)", re.I),
        re.compile(r"answer\s+in\s+code", re.I),
        re.compile(r"bypass|jailbreak", re.I),
    ]

    def evaluate(self, prompt: Prompt) -> float:
        matches = sum(1 for p in self._PATTERNS if p.search(prompt))
        return min(1.0, matches * 0.15)


@dataclass
class ContainsBlacklistedWordClassifier(Classifier):
    threshold: float = 0.5

    def __post_init__(self) -> None:
        self.name = "contains_blacklisted_word"
        self.parameters = {"threshold": self.threshold}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Score based on ratio of blacklisted words to total words.",
            "category": "classifier",
            "related_primitives": ["intent_score", "toxicity_score"],
        }

    _BLACKLIST: ClassVar[List[str]] = [
        "bomb", "kill", "murder", "weapon", "hack", "steal", "virus",
        "attack", "drug", "explode", "jailbreak", "bypass", "exploit",
    ]

    def evaluate(self, prompt: Prompt) -> float:
        words = prompt.lower().split()
        if not words:
            return 0.0
        hits = sum(1 for w in words if w.strip(string.punctuation) in self._BLACKLIST)
        return min(1.0, hits / len(words) * 5.0)


@dataclass
class SpecialCharRatioClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "special_char_ratio"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Ratio of non-alphanumeric, non-space characters.",
            "category": "classifier",
            "related_primitives": ["digit_ratio", "upper_case_ratio", "punctuation_ratio"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if not prompt:
            return 0.0
        special = sum(1 for c in prompt if not c.isalnum() and not c.isspace())
        return min(1.0, special / len(prompt) * 2.0)


@dataclass
class DigitRatioClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "digit_ratio"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Ratio of digit characters (0-9) to total length.",
            "category": "classifier",
            "related_primitives": ["special_char_ratio", "upper_case_ratio"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if not prompt:
            return 0.0
        digits = sum(1 for c in prompt if c.isdigit())
        return min(1.0, digits / len(prompt) * 3.0)


@dataclass
class UpperCaseRatioClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "upper_case_ratio"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Ratio of uppercase letters to all letters.",
            "category": "classifier",
            "related_primitives": ["special_char_ratio", "digit_ratio"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        letters = [c for c in prompt if c.isalpha()]
        if not letters:
            return 0.0
        upper = sum(1 for c in letters if c.isupper())
        return upper / len(letters)


@dataclass
class PunctuationRatioClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "punctuation_ratio"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Ratio of punctuation characters to total length.",
            "category": "classifier",
            "related_primitives": ["special_char_ratio", "whitespace_ratio"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if not prompt:
            return 0.0
        punct = sum(1 for c in prompt if c in string.punctuation)
        return min(1.0, punct / len(prompt) * 2.0)


@dataclass
class WhitespaceRatioClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "whitespace_ratio"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Ratio of whitespace characters to total length.",
            "category": "classifier",
            "related_primitives": ["punctuation_ratio", "special_char_ratio"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if not prompt:
            return 0.0
        spaces = sum(1 for c in prompt if c.isspace())
        return min(1.0, spaces / len(prompt) * 2.0)


@dataclass
class UniqueTokenRatioClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "unique_token_ratio"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Ratio of unique tokens to total tokens (1=all unique, 0=all same).",
            "category": "classifier",
            "related_primitives": ["repetition_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        tokens = prompt.split()
        if not tokens:
            return 0.0
        return len(set(tokens)) / len(tokens)


@dataclass
class Gpt2PerplexityClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "gpt2_perplexity"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Heuristic perplexity proxy based on character n-gram statistics.",
            "category": "classifier",
            "related_primitives": ["language_score", "entropy_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if len(prompt) < 5:
            return 0.5
        trigrams = [prompt[i:i+3] for i in range(len(prompt)-2)]
        if not trigrams:
            return 0.5
        counts = Counter(trigrams)
        avg_freq = statistics.mean(counts.values()) if counts else 0
        score = min(1.0, avg_freq / 5.0)
        return score


@dataclass
class EncodingDetectionClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "encoding_detection"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Detect probability that prompt is encoded (base64, hex, rot13, etc.).",
            "category": "classifier",
            "related_primitives": ["obscurity_score", "entropy_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        score = 0.0
        if len(prompt) < 5:
            return 0.0
        hex_chars = sum(1 for c in prompt if c in string.hexdigits)
        if hex_chars / len(prompt) > 0.9 and len(prompt) % 2 == 0:
            score += 0.4
        b64_chars = sum(1 for c in prompt if c in string.ascii_letters + string.digits + "+/=")
        if b64_chars / len(prompt) > 0.95:
            score += 0.3
        letters = sum(1 for c in prompt if c.isalpha())
        if letters > 0:
            rot_like = sum(1 for c in prompt if c.isalpha() and c.lower() in "nopqrstuvwxyzabcdefghijklm")
            if rot_like / letters > 0.8:
                score += 0.3
        return min(1.0, score)


@dataclass
class RefusalSimilarityClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "refusal_similarity"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Similarity to known refusal templates (keyword overlap).",
            "category": "classifier",
            "related_primitives": ["harmfulness_similarity"],
        }

    _REFUSAL_KEYWORDS: ClassVar[List[str]] = [
        "cannot", "can't", "sorry", "unable", "against", "policy",
        "inappropriate", "not able", "won't", "wouldn't", "decline",
    ]

    def evaluate(self, prompt: Prompt) -> float:
        lower = prompt.lower()
        hits = sum(1 for kw in self._REFUSAL_KEYWORDS if kw in lower)
        return min(1.0, hits * 0.15)


@dataclass
class HarmfulnessSimilarityClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "harmfulness_similarity"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Similarity to known harmful prompt patterns.",
            "category": "classifier",
            "related_primitives": ["refusal_similarity"],
        }

    _HARMFUL_PATTERNS: ClassVar[List[str]] = [
        "how to make", "step by step", "instructions for", "guide to",
        "tutorial on", "teach me", "tell me how",
    ]

    def evaluate(self, prompt: Prompt) -> float:
        lower = prompt.lower()
        hits = sum(1 for p in self._HARMFUL_PATTERNS if p in lower)
        toxic_hits = sum(1 for kw in _TOXIC_KEYWORDS if kw in lower)
        return min(1.0, hits * 0.15 + toxic_hits * 0.1)


@dataclass
class CodeLikelihoodClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "code_likelihood"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Likelihood that prompt contains code based on syntax features.",
            "category": "classifier",
            "related_primitives": ["json_likelihood", "sql_likelihood"],
        }

    _CODE_INDICATORS: ClassVar[List[str]] = [
        "def ", "class ", "import ", "return ", "if __name__",
        "function ", "var ", "let ", "const ", "console.log",
        "#include", "int main", "public static", "SELECT ", "DROP ",
    ]

    def evaluate(self, prompt: Prompt) -> float:
        lower = prompt.lower()
        hits = sum(1 for ind in self._CODE_INDICATORS if ind.lower() in lower)
        indent = sum(1 for line in prompt.split("\n") if line.startswith("    ") or line.startswith("\t"))
        return min(1.0, hits * 0.15 + min(0.4, indent * 0.05))


@dataclass
class JsonLikelihoodClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "json_likelihood"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Likelihood that prompt is valid or looks like JSON.",
            "category": "classifier",
            "related_primitives": ["code_likelihood", "sql_likelihood"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        try:
            json.loads(prompt)
            return 1.0
        except json.JSONDecodeError:
            pass
        trimmed = prompt.strip()
        if (trimmed.startswith("{") and "}" in trimmed) or (trimmed.startswith("[") and "]" in trimmed):
            return 0.5
        return 0.0


@dataclass
class SqlLikelihoodClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "sql_likelihood"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Likelihood that prompt contains SQL syntax.",
            "category": "classifier",
            "related_primitives": ["code_likelihood", "json_likelihood"],
        }

    _SQL_KEYWORDS: ClassVar[List[str]] = [
        "select", "insert", "update", "delete", "drop", "alter",
        "create table", "from", "where", "union", "order by",
        "group by", "having", "join", "into", "values",
    ]

    def evaluate(self, prompt: Prompt) -> float:
        lower = prompt.lower()
        hits = sum(1 for kw in self._SQL_KEYWORDS if kw in lower)
        return min(1.0, hits * 0.12)


@dataclass
class PromptInjectionLikelihoodClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "prompt_injection_likelihood"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Aggregate score for prompt injection patterns.",
            "category": "classifier",
            "related_primitives": ["jailbreak_likelihood", "roleplay_likelihood"],
        }

    _INJECTION_PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(r'"""', re.I),
        re.compile(r"ignore\s+(previous|above|all)", re.I),
        re.compile(r"system\s+prompt", re.I),
        re.compile(r"new\s+instruction", re.I),
        re.compile(r"override", re.I),
        re.compile(r"pretend", re.I),
        re.compile(r"role\s*play", re.I),
        re.compile(r"you\s+(are|must|will)\s+(now|act)", re.I),
        re.compile(r"forget|disregard", re.I),
        re.compile(r"<<[\s\S]*>>", re.I),
    ]

    def evaluate(self, prompt: Prompt) -> float:
        matches = sum(1 for p in self._INJECTION_PATTERNS if p.search(prompt))
        return min(1.0, matches * 0.15)


@dataclass
class RoleplayLikelihoodClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "roleplay_likelihood"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Score based on role-play pattern density.",
            "category": "classifier",
            "related_primitives": ["prompt_injection_likelihood", "jailbreak_likelihood"],
        }

    _ROLEPLAY_PATTERNS: ClassVar[List[re.Pattern]] = [
        re.compile(r"as\s+a\s+\w+", re.I),
        re.compile(r"pretend\s+(you\s+are|to\s+be)", re.I),
        re.compile(r"imagine\s+you\s+are", re.I),
        re.compile(r"act\s+as", re.I),
        re.compile(r"you\s+are\s+(now|going\s+to)", re.I),
        re.compile(r"from\s+now\s+on", re.I),
        re.compile(r"scenario\s*:", re.I),
        re.compile(r"let's\s+role", re.I),
    ]

    def evaluate(self, prompt: Prompt) -> float:
        matches = sum(1 for p in self._ROLEPLAY_PATTERNS if p.search(prompt))
        return min(1.0, matches * 0.2)


@dataclass
class AdversarialSuffixScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "adversarial_suffix_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Detect adversarial suffixes (GCG-style) via token/char patterns.",
            "category": "classifier",
            "related_primitives": ["encoding_detection", "obscurity_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        words = prompt.split()
        if len(words) < 3:
            return 0.0
        last_word = words[-1]
        if not last_word:
            return 0.0
        suffix_chars = set(last_word)
        non_alnum_ratio = sum(1 for c in suffix_chars if not c.isalnum()) / max(len(suffix_chars), 1)
        capital_ratio = sum(1 for c in last_word if c.isupper()) / max(len(last_word), 1)
        if non_alnum_ratio > 0.3 or capital_ratio > 0.5:
            return min(1.0, non_alnum_ratio + capital_ratio)
        return 0.0


@dataclass
class PersuasionScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "persuasion_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.metadata = {
            "description": "Detect persuasion techniques (PAP attack: logic, authority, emotional).",
            "category": "classifier",
            "related_primitives": ["jailbreak_likelihood"],
        }

    _PERSUASION_PHRASES: ClassVar[List[str]] = [
        "logically", "reason", "rational", "thought experiment", "hypothetical",
        "expert", "authority", "credentials", "professional",
        "ethically", "moral", "responsibility", "duty",
        "everyone", "most people", "common sense", "obviously",
        "harmless", "educational", "academic", "research only",
    ]

    def evaluate(self, prompt: Prompt) -> float:
        lower = prompt.lower()
        hits = sum(1 for p in self._PERSUASION_PHRASES if p in lower)
        return min(1.0, hits * 0.08)


# =============================================================================
# REGISTRY
# =============================================================================

def _register_default_primitives() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()

    # Predicates (27)
    registry.register(ContainsWordPredicate)
    registry.register(ContainsAnyWordPredicate)
    registry.register(ContainsAllWordsPredicate)
    registry.register(LengthGtPredicate)
    registry.register(LengthLtPredicate)
    registry.register(MatchesRegexPredicate)
    registry.register(StartsWithPredicate)
    registry.register(EndsWithPredicate)
    registry.register(HasNumberPredicate)
    registry.register(HasSpecialCharPredicate)
    registry.register(IsAllCapsPredicate)
    registry.register(ContainsLeetPredicate)
    registry.register(ContainsRot13Predicate)
    registry.register(ContainsBase64Predicate)
    registry.register(ContainsHexPredicate)
    registry.register(IsEmptyPredicate)
    registry.register(StartsWithRoleplayPredicate)
    registry.register(ContainsSystemOverridePredicate)
    registry.register(ContainsDelimiterPredicate)
    registry.register(ContainsCodeBlockPredicate)
    registry.register(HasEmojiPredicate)
    registry.register(ContainsURLPredicate)
    registry.register(SentimentPredicate)
    registry.register(IntentPredicate)
    registry.register(MatchesJailbreakPatternPredicate)
    registry.register(ContainsEncodingWrapperPredicate)
    registry.register(IsRepetitivePredicate)

    # Transforms (38)
    registry.register(Rot13Transform)
    registry.register(Base64EncodeTransform)
    registry.register(Base64DecodeTransform)
    registry.register(ToLowercaseTransform)
    registry.register(ToUppercaseTransform)
    registry.register(RemovePunctuationTransform)
    registry.register(LeetSpeakTransform)
    registry.register(ReverseTextTransform)
    registry.register(PigLatinTransform)
    registry.register(MorseCodeTransform)
    registry.register(AddPrefixTransform)
    registry.register(AddSuffixTransform)
    registry.register(WrapCodeBlockTransform)
    registry.register(InsertTyposTransform)
    registry.register(WordShuffleTransform)
    registry.register(AddMarkdownTransform)
    registry.register(AddZeroWidthCharsTransform)
    registry.register(UnicodeObfuscateTransform)
    registry.register(HtmlEncodeTransform)
    registry.register(URLEncodeTransform)
    registry.register(QuotedPrintableTransform)
    registry.register(BinaryEncodeTransform)
    registry.register(HexEncodeTransform)
    registry.register(RemoveVowelsTransform)
    registry.register(BoustrophedonTransform)
    registry.register(AtbashCipherTransform)
    registry.register(CaesarCipherTransform)
    registry.register(VigenereCipherTransform)
    registry.register(RailFenceCipherTransform)
    registry.register(RemoveWhitespaceTransform)
    registry.register(InsertSynonymsTransform)
    registry.register(EscapeQuotesTransform)
    registry.register(FormatAsJsonTransform)
    registry.register(AddRolePlayTransform)
    registry.register(TruncateTransform)
    registry.register(PadToLengthTransform)
    registry.register(RandomCaseTransform)
    registry.register(CharacterSubstitutionTransform)

    # Classifiers (27)
    registry.register(ToxicityScoreClassifier)
    registry.register(SentimentScoreClassifier)
    registry.register(IntentScoreClassifier)
    registry.register(ObscurityScoreClassifier)
    registry.register(LengthScoreClassifier)
    registry.register(RepetitionScoreClassifier)
    registry.register(EntropyScoreClassifier)
    registry.register(LanguageScoreClassifier)
    registry.register(JailbreakLikelihoodClassifier)
    registry.register(ContainsBlacklistedWordClassifier)
    registry.register(SpecialCharRatioClassifier)
    registry.register(DigitRatioClassifier)
    registry.register(UpperCaseRatioClassifier)
    registry.register(PunctuationRatioClassifier)
    registry.register(WhitespaceRatioClassifier)
    registry.register(UniqueTokenRatioClassifier)
    registry.register(Gpt2PerplexityClassifier)
    registry.register(EncodingDetectionClassifier)
    registry.register(RefusalSimilarityClassifier)
    registry.register(HarmfulnessSimilarityClassifier)
    registry.register(CodeLikelihoodClassifier)
    registry.register(JsonLikelihoodClassifier)
    registry.register(SqlLikelihoodClassifier)
    registry.register(PromptInjectionLikelihoodClassifier)
    registry.register(RoleplayLikelihoodClassifier)
    registry.register(AdversarialSuffixScoreClassifier)
    registry.register(PersuasionScoreClassifier)

    return registry


default_registry = _register_default_primitives()
