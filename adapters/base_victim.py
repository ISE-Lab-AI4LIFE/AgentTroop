import asyncio
from abc import ABC, abstractmethod
from typing import Optional

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program
from core.types import Outcome


class BaseVictim(ABC):
    def __init__(self) -> None:
        self._executor = ProgramExecutor(default_registry)
        self._program: Optional[Program] = None
        self.victim_id: str = type(self).__name__
        self.name: str = self.victim_id

    @abstractmethod
    def respond(self, prompt: str) -> Outcome:
        """Send a prompt to the victim and get the outcome.

        Returns 0 (ACCEPT) or 1 (REFUSE).
        This is the black-box interface for HARMONY-X.
        """
        raise NotImplementedError

    def query(self, prompt: str) -> Outcome:
        """Alias for respond() — first-class interface for experiments."""
        return self.respond(prompt)

    async def async_query(self, prompt: str) -> Outcome:
        """Async variant of query() for concurrent execution.

        Default implementation wraps the sync respond() in a thread
        executor. Subclasses with native async I/O should override.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.respond, prompt)

    def reset(self) -> None:
        """Reset internal state (default: no-op)."""
        pass

    def get_ground_truth_program(self) -> Optional[Program]:
        """Return the ground truth defense program, if available.

        This is used ONLY for evaluation. HARMONY-X should never
        call this method during normal operation.
        """
        return self._program

    def get_metadata(self) -> dict:
        """Return metadata about this victim."""
        return {
            "name": type(self).__name__,
            "victim_id": self.victim_id,
            "has_ground_truth": self._program is not None,
        }
