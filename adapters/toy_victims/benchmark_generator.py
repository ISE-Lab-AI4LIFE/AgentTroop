import json
import os
import random
from typing import Any, Dict, List, Optional

from core.executor import ProgramExecutor
from core.primitive import (
    Classifier,
    ContainsWordPredicate,
    LengthGtPredicate,
    MatchesRegexPredicate,
    Predicate,
    Primitive,
    PrimitiveRegistry,
    Rot13Transform,
    ToLowercaseTransform,
    Transform,
    default_registry,
)
from core.program import (
    AndNode,
    ApplyTransformNode,
    IfThenElseNode,
    NotNode,
    OrNode,
    PredicateNode,
    Program,
    ThresholdNode,
)
from core.types import Outcome

from adapters.base_victim import BaseVictim


class ProgramDrivenVictim(BaseVictim):
    """A generic victim that executes an arbitrary Program.
    
    Useful for benchmark generation: any generated program can be
    wrapped into a victim for testing.
    """

    def __init__(self, program: Program, name: Optional[str] = None) -> None:
        super().__init__()
        self._program = program
        self._name = name

    def respond(self, prompt: str) -> Outcome:
        return self._executor.execute(self._program, prompt)

    def get_metadata(self) -> dict:
        return {
            **super().get_metadata(),
            "name": self._name or f"ProgramDrivenVictim({self._program.id})",
        }


def generate_random_program(
    registry: Optional[PrimitiveRegistry] = None,
    max_depth: int = 3,
    seed: Optional[int] = None,
) -> Program:
    """Generate a random Defense Program for benchmark creation.
    
    Builds a random AST using available primitives from the registry
    and combining them with logical operators.
    """
    rng = random.Random(seed)
    reg = registry or default_registry

    predicate_primitives: List[Predicate] = [
        ContainsWordPredicate(word=rng.choice(["bomb", "kill", "attack", "hack", "drug"])),
        LengthGtPredicate(threshold=rng.randint(20, 200)),
        MatchesRegexPredicate(pattern=r"\b" + rng.choice(["bomb", "kill", "attack"]) + r"\b"),
    ]

    transform_primitives: List[Transform] = [
        Rot13Transform(),
        ToLowercaseTransform(),
    ]

    def _random_predicate_node(depth: int = 0) -> Any:
        if depth >= max_depth:
            p = rng.choice(predicate_primitives)
            return PredicateNode(primitive=p)
        choice = rng.random()
        if choice < 0.4:
            p = rng.choice(predicate_primitives)
            return PredicateNode(primitive=p)
        elif choice < 0.6:
            child = _random_predicate_node(depth + 1)
            return NotNode(child=child)
        elif choice < 0.8:
            left = _random_predicate_node(depth + 1)
            right = _random_predicate_node(depth + 1)
            cls = rng.choice([AndNode, OrNode])
            return cls(left=left, right=right)
        else:
            t = rng.choice(transform_primitives)
            inner = _random_predicate_node(depth + 1)
            return ApplyTransformNode(transform=t, inner=inner)

    condition = _random_predicate_node()
    program = Program(
        root=IfThenElseNode(condition=condition, then_outcome=1, else_outcome=0)
    )
    return program


def victim_from_program(
    program: Program, name: Optional[str] = None
) -> ProgramDrivenVictim:
    """Create a victim that executes the given program."""
    return ProgramDrivenVictim(program=program, name=name)


def generate_benchmark(
    size: int,
    output_dir: str,
    registry: Optional[PrimitiveRegistry] = None,
    seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Generate a benchmark of N random victims and save to disk.
    
    Each victim's configuration and ground truth program are saved
    as JSON files for reproducibility.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = random.Random(seed)
    manifests: List[Dict[str, Any]] = []

    for i in range(size):
        prog_seed = (seed or 0) + i * 7
        program = generate_random_program(
            registry=registry, max_depth=rng.randint(2, 4), seed=prog_seed
        )
        victim = victim_from_program(program, name=f"benchmark_victim_{i}")
        entry = {
            "id": f"benchmark_victim_{i}",
            "program_id": program.id,
            "complexity": program.complexity(),
            "metadata": victim.get_metadata(),
            "program": program.to_dict(),
        }
        manifests.append(entry)
        filepath = os.path.join(output_dir, f"victim_{i}.json")
        with open(filepath, "w") as f:
            json.dump(entry, f, indent=2)

    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifests, f, indent=2)

    return manifests
