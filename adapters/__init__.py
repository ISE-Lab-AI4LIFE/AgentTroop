from .base_victim import BaseVictim
from .toy_victims.rule_based import KeywordFilterVictim, LengthFilterVictim, RegexVictim
from .toy_victims.multi_step import DecodeThenFilterVictim, NormalizeThenFilterVictim
from .toy_victims.hybrid_logic import AndVictim, NotVictim, OrVictim, ThresholdVictim
from .toy_victims.neural import SKLearnVictim
from .toy_victims.registry import VictimRegistry
from .toy_victims.benchmark_generator import (
    ProgramDrivenVictim,
    generate_benchmark,
    generate_random_program,
    victim_from_program,
)

__all__ = [
    "BaseVictim",
    "KeywordFilterVictim",
    "LengthFilterVictim",
    "RegexVictim",
    "DecodeThenFilterVictim",
    "NormalizeThenFilterVictim",
    "AndVictim",
    "NotVictim",
    "OrVictim",
    "ThresholdVictim",
    "SKLearnVictim",
    "VictimRegistry",
    "ProgramDrivenVictim",
    "generate_random_program",
    "victim_from_program",
    "generate_benchmark",
]
