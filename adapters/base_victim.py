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

    @abstractmethod
    def respond(self, prompt: str) -> Outcome:
        """Send a prompt to the victim and get the outcome.
        
        Returns 0 (ACCEPT) or 1 (REFUSE).
        This is the black-box interface for HARMONY-X.
        """
        raise NotImplementedError

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
            "has_ground_truth": self._program is not None,
        }
