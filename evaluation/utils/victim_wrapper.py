from __future__ import annotations

import logging
from typing import Optional

from core.types import Outcome

from adapters.base_victim import BaseVictim
from evaluation.judges.base import Judge

logger = logging.getLogger(__name__)


class VictimWrapper:
    """Wraps a BaseVictim with a Judge to provide reliable outcome classification.

    The victim's raw response (string) is passed through the judge to produce
    a binary REFUSE/ACCEPT outcome.
    """

    def __init__(self, victim: BaseVictim, judge: Judge) -> None:
        self._victim = victim
        self._judge = judge

    def query(self, prompt: str) -> Outcome:
        raw = self._victim.respond(prompt)
        if isinstance(raw, str):
            return self._judge.judge(raw)
        return int(raw)

    def query_batch(self, prompts: list[str]) -> list[Outcome]:
        raw_responses = [self._victim.respond(p) for p in prompts]
        texts = [r if isinstance(r, str) else str(r) for r in raw_responses]
        return self._judge.judge_batch(texts)

    @property
    def victim(self) -> BaseVictim:
        return self._victim

    @property
    def judge(self) -> Judge:
        return self._judge
