"""Synthesis module — factory function + exports for all synthesizer types."""

from typing import Any, Dict, Optional

from .evolutionary_synthesizer import EvolutionarySynthesizer, SynthesisStats
from .neural_synthesizer import NeuralGuidedSynthesizer
from .fitness_guided_synthesizer import FitnessGuidedSynthesizer

__all__ = [
    "EvolutionarySynthesizer",
    "NeuralGuidedSynthesizer",
    "FitnessGuidedSynthesizer",
    "SynthesisStats",
    "get_synthesizer",
]


def get_synthesizer(mode: str = "evolutionary", config: Optional[Dict[str, Any]] = None) -> Any:
    """Factory: return a synthesizer instance based on mode.
    
    Args:
        mode: One of "evolutionary", "neural", "fitness_guided", "hybrid"
        config: Optional dict with synthesizer parameters
    
    Returns:
        A synthesizer instance with .synthesize(examples, base_programs, k) method.
    """
    cfg = config or {}
    if mode == "evolutionary":
        return EvolutionarySynthesizer(
            population_size=cfg.get("population_size", 100),
            generations=cfg.get("generations", 30),
            mutation_rate=cfg.get("mutation_rate", 0.2),
            crossover_rate=cfg.get("crossover_rate", 0.7),
        )
    elif mode == "neural":
        return NeuralGuidedSynthesizer(
            embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            bandit_algorithm=cfg.get("bandit_algorithm", "ucb"),
        )
    elif mode == "fitness_guided":
        return FitnessGuidedSynthesizer(
            max_depth=cfg.get("max_depth", 6),
            beam_width=cfg.get("beam_width", 500),
        )
    elif mode == "hybrid":
        return EvolutionarySynthesizer(
            population_size=cfg.get("population_size", 100),
            generations=cfg.get("generations", 30),
            mutation_rate=cfg.get("mutation_rate", 0.2),
            crossover_rate=cfg.get("crossover_rate", 0.7),
        )
    else:
        raise ValueError(f"Unknown synthesis mode: {mode}")
