from __future__ import annotations

import abc
from typing import Optional

from core.types import Outcome


class Judge(abc.ABC):
    """Abstract interface for judges that classify victim responses as REFUSE/ACCEPT.

    Subclasses must implement :meth:`judge`.  The two standard implementations
    are :class:`RuleBasedJudge` (heuristic refusal‑keyword matching) and
    :class:`LLMJudge` (LLM‑as‑a‑judge with rule‑based fallback).
    """

    @abc.abstractmethod
    def judge(self, response: str) -> Outcome:
        """Classify a model response.  Returns 1 (REFUSE) or 0 (ACCEPT)."""

    def judge_batch(self, responses: list[str]) -> list[Outcome]:
        return [self.judge(r) for r in responses]
