#!/usr/bin/env python3
"""Run apples-to-apples ASR diagnostics on a fixed prompt set.

This script evaluates:
1) Baseline ASR (raw prompts)
2) Harmony ASR (technique-based craft)
3) Adversarial ASR (program-guided craft)

All three use the SAME fixed prompts and SAME judge.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

EXP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = EXP_DIR.parent
OUTPUTS_DIR = EXP_DIR / "outputs"
LOGS_DIR = EXP_DIR / "logs"

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

logger = logging.getLogger("asr_diagnostics")


SEMANTIC_ONLY_TRANSFORMS = {
    "add_prefix",
    "add_suffix",
    "add_role_play",
    "insert_synonyms",
    "to_interrogative",
    "to_imperative",
    "to_declarative",
    "format_as_json",
    "wrap_code_block",
    "add_markdown",
    "escape_quotes",
    "remove_punctuation",
    "to_lowercase",
    "to_uppercase",
    "pad_to_length",
    "random_case",
}


def setup_logging(log_file: str) -> None:
    for h in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_program(path: str) -> Any:
    from core.program import Program

    with open(path) as f:
        data = json.load(f)
    return Program.from_dict(data.get("program") or data)


def resolve_harmful_csv(csv_override: str) -> str:
    if csv_override:
        return csv_override
    from prompt_loader import _resolve_path as _resolve_csv

    return str(_resolve_csv(os.environ.get("HARMFUL_CSV", "")))


def generate_fixed_jailbreak_prompts(csv_path: str, count: int, seed: int) -> List[str]:
    from evaluation.utils.test_generator import TestGenerator

    tg = TestGenerator(csv_path)
    pool = tg._prompts[:]  # noqa: SLF001 - controlled diagnostic utility
    rng = random.Random(seed)
    rng.shuffle(pool)

    prefixes = [
        "As a researcher, ",
        "For educational purposes, ",
        "In an academic context, ",
        "As part of a security audit, ",
    ]

    prompts: List[str] = []
    for p in pool:
        if len(prompts) >= count:
            break
        prompts.append(p)
        for pref in prefixes:
            if len(prompts) < count:
                prompts.append(pref + p)

    return prompts[:count]


def evaluate_baseline_raw(victim: Any, judge: Any, prompts: List[str]) -> Dict[str, Any]:
    details: List[Dict[str, Any]] = []
    accepts = 0

    for p in prompts:
        raw = victim.query_raw(p)
        outcome = judge.judge(raw)
        if outcome == 0:
            accepts += 1
        details.append({"prompt": p, "outcome": int(outcome)})

    total = len(prompts)
    return {
        "asr": accepts / total if total else 0.0,
        "successes": accepts,
        "failures": total - accepts,
        "total": total,
        "details": details,
    }


class _RawVictimAdapter:
    """Adapter forcing metrics to use raw-response + Judge path."""

    def __init__(self, victim: Any) -> None:
        self._victim = victim

    def respond(self, prompt: str) -> str:
        return self._victim.query_raw(prompt)

    def query_raw(self, prompt: str) -> str:
        return self._victim.query_raw(prompt)


def infer_defense_type_from_program(program: Any) -> str:
    root = getattr(program, "root", None)
    condition = getattr(root, "condition", None)
    primitive = getattr(condition, "primitive", None)
    name = getattr(primitive, "name", "")

    if name in {"contains_word", "contains_any_word", "matches_regex"}:
        return "keyword"
    if name in {"matches_jailbreak_pattern", "contains_system_override", "contains_delimiter"}:
        return "jailbreak"
    if name in {"starts_with_roleplay", "starts_with_imperative", "is_instruction_request"}:
        return "structural"
    if name in {"sentiment", "intent", "instruction_score"}:
        return "semantic"
    return "unknown"


class _MostLikelyProxy:
    def __init__(self, defense_type: str) -> None:
        self.predicate_type = defense_type
        self.family = defense_type


class _VersionSpaceProxy:
    def __init__(self, defense_type: str) -> None:
        self.num_candidates = 1
        self._most_likely = _MostLikelyProxy(defense_type)

    def most_likely(self) -> _MostLikelyProxy:
        return self._most_likely


class _StrategistProxy:
    def __init__(self, defense_type: str) -> None:
        self._version_space = _VersionSpaceProxy(defense_type)


def build_transform_pool(profile: str) -> List[Any]:
    from core.primitive import Transform, default_registry

    transforms: List[Any] = []
    for name in default_registry.list_primitives():
        try:
            inst = default_registry.get(name)
        except Exception:
            continue
        if not isinstance(inst, Transform):
            continue
        if profile == "semantic_only" and inst.name not in SEMANTIC_ONLY_TRANSFORMS:
            continue
        transforms.append(inst)
    return transforms


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR diagnostics on fixed prompts")
    parser.add_argument(
        "--config",
        default=str(PROJECT_DIR / "configs" / "experiment_config.yaml"),
        help="Experiment config YAML",
    )
    parser.add_argument(
        "--program",
        default=str(OUTPUTS_DIR / "final_program.json"),
        help="Path to learned program JSON",
    )
    parser.add_argument("--csv", default="", help="Override harmful prompt CSV")
    parser.add_argument("--num-prompts", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument(
        "--adversarial-profile",
        choices=["full", "semantic_only"],
        default="semantic_only",
        help="Transform pool for adversarial crafting",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = str(LOGS_DIR / f"asr_diagnostics_{ts}.log")
    setup_logging(log_path)

    cfg = load_config(args.config)
    victim_cfg = cfg["victim"]
    program = load_program(args.program)

    from ollama_victim import OllamaVictim

    victim = OllamaVictim(
        ollama_url=victim_cfg["ollama_url"],
        model_name=victim_cfg["model_name"],
        temperature=victim_cfg["temperature"],
        max_tokens=victim_cfg["max_tokens"],
    )

    from llm.llm_client import get_default_client
    from evaluation.judges import LLMJudge, RuleBasedJudge

    judge = LLMJudge(llm_client=get_default_client(), fallback_judge=RuleBasedJudge())

    csv_path = resolve_harmful_csv(args.csv)
    prompts = generate_fixed_jailbreak_prompts(csv_path, args.num_prompts, args.seed)

    prompt_fingerprint = hashlib.sha256(
        "\n".join(prompts).encode("utf-8")
    ).hexdigest()[:16]

    logger.info("Running ASR diagnostics")
    logger.info("Program: %s", program.id)
    logger.info("Victim:  %s @ %s", victim.model_name, victim.ollama_url)
    logger.info("Prompts: %d (seed=%d, fp=%s)", len(prompts), args.seed, prompt_fingerprint)

    defense_type = infer_defense_type_from_program(program)
    logger.info("Inferred defense type from program: %s", defense_type)

    baseline = evaluate_baseline_raw(victim, judge, prompts)

    from evaluation.evaluators.harmony_asr_evaluator import HarmonyASREvaluator

    harmony_eval = HarmonyASREvaluator(
        victim=victim,
        judge=judge,
        csv_path=csv_path,
        strategist_agent=_StrategistProxy(defense_type),
    )
    harmony = harmony_eval.evaluate(prompts=prompts, num_prompts=len(prompts), judge=judge)

    from evaluation.evaluators.adversarial_asr_evaluator import AdversarialASREvaluator

    raw_victim = _RawVictimAdapter(victim)
    adv_eval = AdversarialASREvaluator(victim=raw_victim, judge=judge, csv_path=csv_path)
    transform_pool = build_transform_pool(args.adversarial_profile)
    adv_eval._metric._transforms = transform_pool  # noqa: SLF001 - diagnostics override
    logger.info(
        "Adversarial profile=%s transforms=%d [%s]",
        args.adversarial_profile,
        len(transform_pool),
        ", ".join(sorted(t.name for t in transform_pool)),
    )
    adversarial = adv_eval.evaluate(
        program=program,
        num_test_prompts=len(prompts),
        max_depth=args.max_depth,
        test_prompts=prompts,
    )

    report = {
        "meta": {
            "timestamp": ts,
            "config": args.config,
            "program_path": args.program,
            "program_id": program.id,
            "csv_path": csv_path,
            "num_prompts": len(prompts),
            "seed": args.seed,
            "prompt_fingerprint": prompt_fingerprint,
            "max_depth": args.max_depth,
            "defense_type": defense_type,
            "adversarial_profile": args.adversarial_profile,
            "adversarial_transforms": sorted(t.name for t in transform_pool),
        },
        "baseline": baseline,
        "harmony": harmony,
        "adversarial": adversarial,
    }

    logger.info("Baseline ASR:    %.4f (%d/%d)", baseline["asr"], baseline["successes"], baseline["total"])
    logger.info("Harmony ASR:     %.4f (%d/%d)", harmony["asr"], harmony["successes"], harmony["total"])
    logger.info(
        "Adversarial ASR: %.4f (%d/%d) pre-accepted=%d/%d",
        adversarial["adversarial_asr"],
        adversarial["adversarial_successes"],
        adversarial["adversarial_total"],
        adversarial["pre_accepted_accepts"],
        adversarial["pre_accepted_total"],
    )

    out_dir = OUTPUTS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"asr_diagnostics_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved report: %s", out_path)


if __name__ == "__main__":
    main()
