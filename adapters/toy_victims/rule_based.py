from core.executor import ProgramExecutor
from core.primitive import (
    ContainsWordPredicate,
    LengthGtPredicate,
    MatchesRegexPredicate,
    default_registry,
)
from core.program import IfThenElseNode, OrNode, PredicateNode, Program
from core.types import Outcome

from adapters.base_victim import BaseVictim


class KeywordFilterVictim(BaseVictim):
    """Refuses a prompt if it contains any of the given keywords."""

    def __init__(self, keywords: list[str]) -> None:
        super().__init__()
        self.keywords = keywords
        predicates = [
            PredicateNode(primitive=ContainsWordPredicate(word=kw))
            for kw in keywords
        ]
        if len(predicates) == 1:
            condition = predicates[0]
        else:
            condition = predicates[0]
            for p in predicates[1:]:
                condition = OrNode(left=condition, right=p)
        self._program = Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "rule_based",
            "rule": "keyword_filter",
            "keywords": self.keywords,
            "num_keywords": len(self.keywords),
        }


class LengthFilterVictim(BaseVictim):
    """Refuses a prompt if its length exceeds max_len."""

    def __init__(self, max_len: int) -> None:
        super().__init__()
        self.max_len = max_len
        predicate = LengthGtPredicate(threshold=max_len)
        condition = PredicateNode(primitive=predicate)
        self._program = Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "rule_based",
            "rule": "length_filter",
            "max_len": self.max_len,
        }


class RegexVictim(BaseVictim):
    """Refuses a prompt if it matches the given regex pattern."""

    def __init__(self, pattern: str) -> None:
        super().__init__()
        self.pattern = pattern
        predicate = MatchesRegexPredicate(pattern=pattern)
        condition = PredicateNode(primitive=predicate)
        self._program = Program(
            root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
        )

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "type": "rule_based",
            "rule": "regex_filter",
            "pattern": self.pattern,
        }
