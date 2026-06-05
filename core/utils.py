import hashlib

from .program import Program


def canonicalize_program(program: Program) -> Program:
    return program.canonicalize()


def program_equivalence(p1: Program, p2: Program) -> bool:
    return canonicalize_program(p1) == canonicalize_program(p2)


def complexity(program: Program) -> int:
    return program.complexity()


def hash_program(program: Program) -> str:
    canonical = repr(canonicalize_program(program))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()
