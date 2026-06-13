"""Thin re-export of OpenRouterVictim for the llama-3-8b-instruct experiment.

Keeps the experiment directory self-contained while sharing the actual
implementation with other experiments via ``adapters.openrouter_victim``.
"""

from adapters.openrouter_victim import OpenRouterVictim

__all__ = ["OpenRouterVictim"]
