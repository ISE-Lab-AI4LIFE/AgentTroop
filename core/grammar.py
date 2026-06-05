from typing import List

from .primitive import PrimitiveRegistry


class Grammar:
    def __init__(self, registry: PrimitiveRegistry) -> None:
        self.registry = registry

    def get_primitive_names(self) -> List[str]:
        return self.registry.list_primitives()

    def to_smtlib(self) -> str:
        primitive_names = self.get_primitive_names()
        grammar_lines = [
            ";; SMT-LIB grammar for Defense Program synthesis",
            "(set-logic ALL)",
            "(declare-fun prompt () String)",
            "(declare-fun output () Int)",
            "(declare-fun refuse () Bool)",
            "(define-fun is_refuse ((p String)) Bool true)",
            ";; Primitive syntax templates",
        ]
        for name in primitive_names:
            grammar_lines.append(f";; primitive: {name}")
            grammar_lines.append(f"(declare-fun {name} ((String)) Bool)")
        grammar_lines.extend([
            "(define-fun program ((p String)) Bool",
            "  (or",
            *[f"    ({name} p)" for name in primitive_names],
            "  )",
            ")",
            "(assert (program prompt))",
            "(check-sat)",
        ])
        return "\n".join(grammar_lines)
