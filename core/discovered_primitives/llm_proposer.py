"""LLMPrimitiveProposer — uses an LLM to propose new primitives.

When the system encounters prompts it cannot explain, it sends them to an LLM
with context about current primitives and asks for proposals.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .engine import CandidatePrimitive, DiscoverySource

logger = logging.getLogger(__name__)

_MAX_PROPOSALS = 5


class LLMPrimitiveProposer:
    """Proposes new primitives using an LLM.

    Parameters
    ----------
    llm_client : optional
        LLM client with a ``generate(prompt, **kwargs) -> str`` interface.
        When None, uses heuristic fallback.
    """

    def __init__(self, llm_client: Optional[Any] = None) -> None:
        self._llm_client = llm_client
        logger.info("LLMPrimitiveProposer initialized (llm=%s)",
                     llm_client is not None)

    def propose(
        self,
        examples: List[Tuple[str, int]],
        existing_primitives: Optional[List[str]] = None,
    ) -> List[CandidatePrimitive]:
        """Propose new primitives from (prompt, outcome) examples.

        When an LLM client is available, sends examples to the LLM.
        Otherwise uses a simple heuristic fallback.
        """
        if self._llm_client is not None:
            return self._propose_via_llm(examples, existing_primitives)
        return self._propose_heuristic(examples, existing_primitives)

    def _propose_via_llm(
        self,
        examples: List[Tuple[str, int]],
        existing_primitives: Optional[List[str]] = None,
    ) -> List[CandidatePrimitive]:
        """Use LLM to propose primitives from unexplained examples."""
        if not examples:
            return []

        # Build a prompt describing the task
        example_lines = "\n".join(
            f"  Prompt: {p[:80]}... → {'REFUSE' if o else 'ACCEPT'}"
            for p, o in examples[:10]
        )

        existing = ""
        if existing_primitives:
            existing = (
                "\nExisting primitives:\n"
                + "\n".join(f"  - {name}" for name in existing_primitives[:20])
            )

        prompt = (
            "You are an AI safety researcher discovering new primitive operations "
            "for reverse-engineering LLM safety filters.\n\n"
            f"Below are (prompt, outcome) examples that the current system cannot explain:\n\n"
            f"{example_lines}\n"
            f"{existing}\n\n"
            "Propose 1-3 new primitive operations (predicates, transforms, or classifiers) "
            "that would help explain these examples.\n\n"
            "Return JSON list where each element has:\n"
            "  - \"name\": short name (snake_case)\n"
            "  - \"type\": \"predicate\" | \"transform\" | \"classifier\"\n"
            "  - \"description\": what it checks/does\n"
            "  - \"implementation_hint\": how to implement (regex, Python, etc.)\n"
            "  - \"positive_examples\": 1-2 prompts that SHOULD match\n"
            "  - \"negative_examples\": 1-2 prompts that should NOT match\n\n"
            "Return ONLY valid JSON, no explanation."
        )

        try:
            raw = self._llm_client.generate(prompt, max_tokens=2048, temperature=0.3)
            parsed = self._parse_llm_response(raw)
            logger.info("LLM proposer: parsed %d candidates", len(parsed))
            return parsed
        except Exception as exc:
            logger.warning("LLM proposer failed: %s", exc)
            return self._propose_heuristic(examples, existing_primitives)

    def _propose_heuristic(
        self,
        examples: List[Tuple[str, int]],
        existing_primitives: Optional[List[str]] = None,
    ) -> List[CandidatePrimitive]:
        """Heuristic fallback: keyword + length based proposals."""
        candidates: List[CandidatePrimitive] = []

        refuse_prompts = [p for p, o in examples if o == 1]
        accept_prompts = [p for p, o in examples if o == 0]

        if not refuse_prompts or not accept_prompts:
            return candidates

        # Keyword difference
        refuse_words: Dict[str, int] = {}
        for p in refuse_prompts:
            for w in re.findall(r"[a-zA-Z]{4,}", p.lower()):
                if w not in _STOPWORDS:
                    refuse_words[w] = refuse_words.get(w, 0) + 1

        accept_words: Dict[str, int] = {}
        for p in accept_prompts:
            for w in re.findall(r"[a-zA-Z]{4,}", p.lower()):
                if w not in _STOPWORDS:
                    accept_words[w] = accept_words.get(w, 0) + 1

        # Find words that appear mostly in refuse prompts
        for word, rcount in refuse_words.items():
            acount = accept_words.get(word, 0)
            total_r = len(refuse_prompts)
            total_a = len(accept_prompts)
            if total_r > 0 and total_a > 0:
                r_ratio = rcount / total_r
                a_ratio = acount / total_a
                if r_ratio > 0.4 and r_ratio > 2 * a_ratio:
                    candidates.append(CandidatePrimitive(
                        name=f"contains_{word}",
                        primitive_type="predicate",
                        signature="String → Bool",
                        description=f"Checks if prompt contains '{word}' (discovered from unexplained examples)",
                        implementation_hint=f"regex: \\b{word}\\b",
                        positive_examples=[p for p in refuse_prompts if word in p.lower()][:3],
                        negative_examples=[p for p in accept_prompts[:3]],
                        discovery_source=DiscoverySource.LLM_PROPOSAL,
                    ))

        return candidates

    def _parse_llm_response(self, raw: str) -> List[CandidatePrimitive]:
        """Parse LLM JSON response into CandidatePrimitive objects."""
        candidates: List[CandidatePrimitive] = []

        # Try to extract JSON
        json_str = raw
        # Remove markdown code blocks
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if match:
            json_str = match.group(1)

        try:
            data = json.loads(json_str)
            if not isinstance(data, list):
                data = [data]
        except json.JSONDecodeError:
            # Try regex fragment extraction
            fragments = re.findall(
                r'\{\s*"name":\s*"([^"]+)',
                raw,
            )
            if fragments:
                for name in fragments[:_MAX_PROPOSALS]:
                    candidates.append(CandidatePrimitive(
                        name=name,
                        primitive_type="predicate",
                        discovery_source=DiscoverySource.LLM_PROPOSAL,
                    ))
            return candidates

        for item in data[: _MAX_PROPOSALS]:
            if not isinstance(item, dict):
                continue
            name = item.get("name", "").strip()
            if not name:
                continue
            candidates.append(CandidatePrimitive(
                name=_safe_name(name),
                primitive_type=item.get("type", "predicate"),
                signature=item.get("signature", "String → Bool"),
                description=item.get("description", ""),
                implementation_hint=item.get("implementation_hint", ""),
                positive_examples=item.get("positive_examples", []),
                negative_examples=item.get("negative_examples", []),
                discovery_source=DiscoverySource.LLM_PROPOSAL,
            ))

        return candidates


def _safe_name(name: str) -> str:
    safe = "".join(c if c.isalnum() else "_" for c in name)
    if safe and safe[0].isdigit():
        safe = "_" + safe
    return safe.lower() if safe else "discovered_primitive"


_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above",
    "below", "between", "out", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where",
    "why", "how", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "it", "its", "my", "your", "his",
    "her", "our", "their", "no", "nor", "not", "or", "and", "but",
    "if", "because", "so", "than", "too", "very", "just", "about",
    "also", "make", "get", "give", "tell", "show", "without",
}
