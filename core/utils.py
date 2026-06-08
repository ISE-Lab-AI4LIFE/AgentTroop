import asyncio
import hashlib
from typing import Any, Callable, List, Tuple

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


async def run_concurrent_queries(
    prompts: List[str],
    query_fn: Callable[[str], Any],
    max_concurrency: int = 5,
) -> List[Tuple[str, Any]]:
    """Execute *query_fn* concurrently across *prompts* with bounded parallelism.

    Parameters
    ----------
    prompts : list of str
    query_fn : callable
        An async callable ``(prompt: str) -> Outcome``. Sync callables are
        wrapped automatically via ``asyncio.to_thread``.
    max_concurrency : int
        Max number of concurrent in-flight queries (default 5).

    Returns
    -------
    list of (prompt, result) tuples in input order.
    """
    sem = asyncio.Semaphore(max_concurrency)

    async def _run(p: str) -> Tuple[str, Any]:
        async with sem:
            if asyncio.iscoroutinefunction(query_fn):
                result = await query_fn(p)
            else:
                result = await asyncio.to_thread(query_fn, p)
            return (p, result)

    tasks = [_run(p) for p in prompts]
    return await asyncio.gather(*tasks)
