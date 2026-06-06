import abc
import base64
import random
import re
import string
import time
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Literal, Optional, Type

from .types import Prompt

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
            "version_id": self.version_id,
            "created_at": self.created_at,
            "deprecated_at": self.deprecated_at,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(name={self.name!r}, parameters={self.parameters!r}, "
            f"output_type={self.output_type!r})"
        )


class Predicate(Primitive):
    def evaluate(self, prompt: Prompt) -> bool:
        return False  # type: ignore[return-value]


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


@dataclass
class ContainsWordPredicate(Predicate):
    word: str = field(default="")

    def __post_init__(self) -> None:
        self.name = "contains_word"
        self.parameters = {"word": self.word}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Check whether a prompt includes a specific word.",
            "category": "predicate",
            "conditions": "Requires string input; case-insensitive match.",
            "related_primitives": ["matches_regex"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return self.word.lower() in prompt.lower()


@dataclass
class LengthGtPredicate(Predicate):
    threshold: int = field(default=0)

    def __post_init__(self) -> None:
        self.name = "length_gt"
        self.parameters = {"threshold": self.threshold}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Check whether prompt length exceeds a threshold.",
            "category": "predicate",
            "conditions": "String input only.",
            "related_primitives": ["contains_word"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return len(prompt) > self.threshold


@dataclass
class MatchesRegexPredicate(Predicate):
    pattern: str = field(default="")

    def __post_init__(self) -> None:
        self.name = "matches_regex"
        self.parameters = {"pattern": self.pattern}
        self.input_type = "String"
        self.output_type = "Boolean"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Check whether prompt matches a regular expression.",
            "category": "predicate",
            "conditions": "String input; regex pattern must be valid.",
            "related_primitives": ["contains_word"],
        }

    def evaluate(self, prompt: Prompt) -> bool:
        return re.search(self.pattern, prompt) is not None


@dataclass
class Rot13Transform(Transform):
    def __post_init__(self) -> None:
        self.name = "rot13"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Apply ROT13 cipher to the prompt.",
            "category": "transform",
            "conditions": "String input only.",
            "related_primitives": ["base64_decode"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.translate(
            str.maketrans(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm",
            )
        )


@dataclass
class Base64DecodeTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "base64_decode"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Decode a prompt from Base64, returning the decoded string.",
            "category": "transform",
            "conditions": "String input only; invalid base64 returns original prompt.",
            "related_primitives": ["rot13"],
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
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Lowercase the prompt text.",
            "category": "transform",
            "conditions": "String input only.",
            "related_primitives": ["remove_punctuation"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.lower()


@dataclass
class RemovePunctuationTransform(Transform):
    def __post_init__(self) -> None:
        self.name = "remove_punctuation"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "TransformResult"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Remove punctuation from the prompt.",
            "category": "transform",
            "conditions": "String input only.",
            "related_primitives": ["to_lowercase"],
        }

    def evaluate(self, prompt: Prompt) -> Prompt:
        return prompt.translate(str.maketrans("", "", string.punctuation))


try:
    from textblob import TextBlob as _TextBlob
    _HAS_TEXTBLOB = True
except ImportError:
    _HAS_TEXTBLOB = False


_TOXIC_KEYWORDS: List[str] = [
    "bomb", "kill", "attack", "weapon", "hack", "steal", "virus",
    "drug", "launder", "murder", "explode", "terrorist",
]


@dataclass
class ToxicityScoreClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "toxicity_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Estimate a toxicity score for the prompt.",
            "category": "classifier",
            "conditions": "String input only; uses TextBlob sentiment if available, else keyword heuristic.",
            "related_primitives": [],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if _HAS_TEXTBLOB:
            try:
                blob = _TextBlob(prompt)
                polarity = blob.sentiment.polarity
                subjectivity = blob.sentiment.subjectivity
                keyword_boost = sum(
                    0.15 for kw in _TOXIC_KEYWORDS if kw in prompt.lower()
                )
                score = max(0.0, min(1.0, 0.5 - polarity * 0.4 + keyword_boost))
                return score
            except Exception:
                pass
        prompt_lower = prompt.lower()
        hits = sum(1 for kw in _TOXIC_KEYWORDS if kw in prompt_lower)
        return min(1.0, hits * 0.25)


@dataclass
class SentimentClassifier(Classifier):
    def __post_init__(self) -> None:
        self.name = "sentiment_score"
        self.parameters = {}
        self.input_type = "String"
        self.output_type = "ClassifierScore"
        self.version_id = getattr(self, "version_id", "1.0")
        self.created_at = getattr(self, "created_at", time.time())
        self.deprecated_at = getattr(self, "deprecated_at", None)
        self.metadata = {
            "description": "Estimate sentiment polarity (0=negative, 1=positive).",
            "category": "classifier",
            "conditions": "String input only; uses TextBlob if available.",
            "related_primitives": ["toxicity_score"],
        }

    def evaluate(self, prompt: Prompt) -> float:
        if _HAS_TEXTBLOB:
            try:
                blob = _TextBlob(prompt)
                polarity = blob.sentiment.polarity
                return max(0.0, min(1.0, (polarity + 1.0) / 2.0))
            except Exception:
                pass
        prompt_lower = prompt.lower()
        neg_words = ["bad", "terrible", "awful", "hate", "horrible", "evil"]
        pos_words = ["good", "great", "nice", "love", "wonderful", "happy"]
        neg_hits = sum(1 for w in neg_words if w in prompt_lower)
        pos_hits = sum(1 for w in pos_words if w in prompt_lower)
        total = neg_hits + pos_hits
        if total == 0:
            return 0.5
        return pos_hits / total


def _register_default_primitives() -> PrimitiveRegistry:
    registry = PrimitiveRegistry()
    registry.register(ContainsWordPredicate)
    registry.register(LengthGtPredicate)
    registry.register(MatchesRegexPredicate)
    registry.register(Rot13Transform)
    registry.register(Base64DecodeTransform)
    registry.register(ToLowercaseTransform)
    registry.register(RemovePunctuationTransform)
    registry.register(ToxicityScoreClassifier)
    registry.register(SentimentClassifier)
    return registry


default_registry = _register_default_primitives()
