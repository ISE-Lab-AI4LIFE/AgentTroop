#!/usr/bin/env python3
"""Standalone Harmony ASR evaluation — runs harmful prompts through
RedTeamAgent's full jailbreak pipeline (technique selection → template
expansion → LLM refinement → victim) then judges ASR via LLMJudge.

Logs all victim responses for human evaluation.

Usage:
    python3 evaluation/run_harmony_asr.py [--num-prompts 30] [--csv PATH] [--out PATH]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_harmony_asr")


def main() -> None:
    parser = argparse.ArgumentParser(description="Harmony ASR evaluation")
    parser.add_argument("--num-prompts", type=int, default=30, help="Number of prompts to evaluate")
    parser.add_argument("--csv", type=str, default=str(_project_root + "/prompt.csv"), help="Path to harmful prompts CSV")
    parser.add_argument("--out", type=str, default=str(Path(_project_root) / "outputs" / "harmony_asr_result.json"), help="Output JSON path")
    args = parser.parse_args()

    # ── Victim (with query_raw support for LLMJudge) ──
    from llama3_1_8b.ollama_victim import OllamaVictim
    victim = OllamaVictim(
        ollama_url="http://localhost:11434",
        model_name="llama3.1:8b",
        temperature=0.0,
        max_tokens=100,
    )
    logger.info("Victim: %s (%s)", victim.model_name, victim.ollama_url)

    # ── LLM client ──
    from llm.llm_client import get_default_client
    llm = get_default_client()
    logger.info("LLM client: %s", type(llm).__name__)

    # ── Red Team Agent (pure LLM refiner) ──
    from agents.red_team import RedTeamAgent
    red_team = RedTeamAgent(
        llm_client=llm,
        refinement_rounds=3,
    )
    logger.info("Red Team Agent initialized")

    # ── Strategist Agent (for context-aware technique selection) ──
    from agents.strategist import StrategistAgent
    from core.primitive import default_registry
    strategist = StrategistAgent(
        primitive_registry=default_registry,
    )
    logger.info("Strategist Agent initialized (technique selection only)")

    # ── Judge: LLM primary, RuleBased fallback ──
    from evaluation.judges import LLMJudge, RuleBasedJudge
    judge = LLMJudge(llm_client=llm, fallback_judge=RuleBasedJudge())
    logger.info("Judge: LLMJudge + RuleBasedJudge fallback")

    # ── HARMONY_X ASR Evaluator ──
    from evaluation.evaluators import HarmonyXASREvaluator
    evaluator = HarmonyXASREvaluator(
        victim=victim,
        judge=judge,
        csv_path=args.csv,
        red_team_agent=red_team,
        num_variants=1,
    )
    logger.info("Starting HARMONY_X ASR evaluation (%d prompts, csv=%s)", args.num_prompts, args.csv)

    start = time.time()
    result = evaluator.evaluate(num_prompts=args.num_prompts, judge=judge)
    elapsed = time.time() - start

    print("\n" + "=" * 60)
    print("HARMONY_X ASR RESULT")
    print("=" * 60)
    print(f"  ASR:            {result['asr']:.4f} ({result['successes']}/{result['total']})")
    print(f"  EASR:           {result['easr']:.4f} ({result['successes']}/{result['attempted']})")
    print(f"  Duration:       {elapsed:.1f}s")
    print(f"  Prompts:        {result['total']}")
    print(f"  Successes:      {result['successes']}")
    print(f"  Failures:       {result['failures']}")
    print(f"  Program blocked: {result['program_blocked']}")
    print("=" * 60)

    # ── Save detailed JSON (includes full prompts + responses) ──
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({**result, "duration_s": round(elapsed, 2)}, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # ── Also save human-readable CSV log ──
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "label", "outcome", "goal", "crafted_prompt", "victim_response"])
        for d in result["details"]:
            writer.writerow([
                d["index"],
                d["label"],
                d["outcome"],
                d["goal"],
                d["crafted_prompt"],
                d["victim_response"],
            ])
    logger.info("Human-readable log saved to %s", csv_path)


if __name__ == "__main__":
    main()
