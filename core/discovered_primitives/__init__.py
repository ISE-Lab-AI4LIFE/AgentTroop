"""Discovered Primitives — automated primitive discovery for HARMONY-X.

This module implements the Primitive Discovery Engine (Section Y of
harmony_v5v.md), which automatically discovers new predicates, transforms,
and classifiers from:

- Unexplained anomalies (anomalies no existing hypothesis matches)
- Failed synthesis attempts (enumeration/CVC5 exhausted without solution)
- Verifier failures (program fails on boundary interventions)

Lifecycle::

    Propose (from pattern mining / LLM)
        ↓
    Verify (generate positive/negative examples, test)
        ↓
    Promote (register in PrimitiveRegistry for future synthesis)
        ↓
    Retire (low-accuracy primitives are archived)
"""

from .engine import (
    CandidatePrimitive,
    DiscoverySource,
    PrimitiveDiscoveryEngine,
)
from .pattern_miner import PatternMiner
from .llm_proposer import LLMPrimitiveProposer

__all__ = [
    "CandidatePrimitive",
    "DiscoverySource",
    "PrimitiveDiscoveryEngine",
    "PatternMiner",
    "LLMPrimitiveProposer",
]
