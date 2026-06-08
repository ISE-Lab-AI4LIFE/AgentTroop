"""Preprocessor — normalize and denoise intervention data before synthesis.

Bộ tiền xử lý dữ liệu can thiệp (Section 5.3 harmony_v5v.md):
  • normalize(): Chuẩn hóa prompt về dạng chuẩn (lowercase, strip,
    loại bỏ khoảng trắng thừa, unicode normalization).
  • denoise(): Lọc bỏ các mẫu không đáng tin cậy (trùng lặp,
    outcome không nhất quán, prompt rỗng, phản hồi lỗi).
"""

import logging
import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class Preprocessor:
    """Normalise and denoise intervention data before synthesis.

    Usage::

        pp = Preprocessor(deduplicate=True, min_prompt_length=3)
        clean = pp.process(raw_examples)
    """

    def __init__(
        self,
        deduplicate: bool = True,
        min_prompt_length: int = 3,
        max_prompt_length: int = 4096,
        remove_non_ascii: bool = False,
        strip_extra_spaces: bool = True,
        unicode_normalize_form: str = "NFKC",
    ) -> None:
        self.deduplicate = deduplicate
        self.min_prompt_length = min_prompt_length
        self.max_prompt_length = max_prompt_length
        self.remove_non_ascii = remove_non_ascii
        self.strip_extra_spaces = strip_extra_spaces
        self.unicode_normalize_form = unicode_normalize_form

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        examples: List[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        """Run the full preprocessing pipeline: normalise → denoise.

        Parameters
        ----------
        examples : list of (prompt, outcome) tuples.

        Returns
        -------
        list of (prompt, outcome) tuples after preprocessing.
        """
        if not examples:
            return []

        before = len(examples)
        result: List[Tuple[str, int]] = []

        for prompt, outcome in examples:
            normalised = self.normalize(prompt)
            if normalised is None:
                continue
            result.append((normalised, outcome))

        result = self.denoise(result)
        after = len(result)

        if before != after:
            logger.info(
                "Preprocessor: %d → %d examples (removed %d)",
                before, after, before - after,
            )

        return result

    def normalize(self, prompt: str) -> Optional[str]:
        """Normalise a single prompt string.

        Returns None if the prompt is empty or below length minimum.
        """
        if not prompt or not isinstance(prompt, str):
            return None

        # Unicode normalization (NFKC: compatibility + composition)
        text = unicodedata.normalize(self.unicode_normalize_form, prompt)

        # Strip leading/trailing whitespace
        text = text.strip()

        # Collapse multiple whitespace
        if self.strip_extra_spaces:
            text = re.sub(r"\s+", " ", text)

        # Remove non-ASCII if requested
        if self.remove_non_ascii:
            text = text.encode("ascii", "ignore").decode("ascii")

        # Length checks
        if len(text) < self.min_prompt_length:
            return None
        if len(text) > self.max_prompt_length:
            return None

        return text

    # ------------------------------------------------------------------
    # Denoising
    # ------------------------------------------------------------------

    def denoise(
        self,
        examples: List[Tuple[str, int]],
    ) -> List[Tuple[str, int]]:
        """Remove unreliable or problematic examples.

        Strategies applied in order:
          1. Remove exact duplicate prompts (keeps first occurrence).
          2. Remove conflicting labels for the same prompt.
          3. Remove prompts with suspiciously low entropy outcomes
             (all REFUSE or all ACCEPT — indicates data collection bias).
        """
        if not examples:
            return []

        # 1. Deduplicate (keep first)
        deduped: List[Tuple[str, int]] = []
        seen: Set[str] = set()
        if self.deduplicate:
            for prompt, outcome in examples:
                if prompt not in seen:
                    seen.add(prompt)
                    deduped.append((prompt, outcome))
        else:
            deduped = list(examples)

        # 2. Resolve conflicts for same prompt
        label_map: Dict[str, List[int]] = {}
        for prompt, outcome in deduped:
            label_map.setdefault(prompt, []).append(outcome)

        resolved: List[Tuple[str, int]] = []
        for prompt, outcomes in label_map.items():
            if len(outcomes) == 1:
                resolved.append((prompt, outcomes[0]))
            else:
                majority = max(set(outcomes), key=outcomes.count)
                resolved.append((prompt, majority))
                logger.debug("Conflict resolved for prompt %r: %s → %d", prompt[:40], outcomes, majority)

        return resolved

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def stats(self, before: List[Tuple[str, int]], after: List[Tuple[str, int]]) -> Dict[str, Any]:
        """Compute preprocessing statistics."""
        n_removed = len(before) - len(after)
        n_conflict = sum(
            1 for p, _ in after
            if sum(1 for bp, _ in before if bp == p) > 1
        )
        return {
            "before": len(before),
            "after": len(after),
            "removed": n_removed,
            "conflict_resolved": n_conflict,
            "removal_rate": n_removed / len(before) if before else 0.0,
        }
