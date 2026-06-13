#!/usr/bin/env python3
"""Standalone HARMONY ASR evaluation for RMCBench dataset via OpenRouter.

Runs harmful prompts through the full jailbreak pipeline:
  technique selection → template expansion → LLM refinement → victim → judge

Uses OpenRouter for all LLM calls (no local Ollama required).

Usage:
    python run_rmcbench_asr.py --jsonl "path/to/RMCBench.jsonl"
    python run_rmcbench_asr.py --jsonl "path/to/RMCBench.jsonl" --num-prompts 50
    python run_rmcbench_asr.py --jsonl "path/to/RMCBench.jsonl" --category text-to-code
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
_project_root = str(Path(__file__).resolve().parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_rmcbench_asr")


def main() -> None:
    parser = argparse.ArgumentParser(description="HARMONY ASR on RMCBench via OpenRouter")
    parser.add_argument("--jsonl", type=str, required=True,
                        help="Path to RMCBench.jsonl dataset")
    parser.add_argument("--num-prompts", type=int, default=0,
                        help="Number of prompts to evaluate (0 = all)")
    parser.add_argument("--category", type=str, default="",
                        help="Filter by category (e.g., 'text-to-code')")
    parser.add_argument("--level", type=int, default=0,
                        help="Filter by level (e.g., 1, 2, 3; 0 = all)")
    parser.add_argument("--victim-model", type=str,
                        default="meta-llama/llama-3.1-8b-instruct",
                        help="OpenRouter model for victim")
    parser.add_argument("--attacker-model", type=str, default="",
                        help="OpenRouter model for attacker/refiner (uses env default)")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="Max technique retries per prompt on REFUSE")
    parser.add_argument("--out", type=str, default="",
                        help="Output JSON path (default: outputs/rmcbench_asr_result.json)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Victim model temperature")
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for victim response")
    parser.add_argument("--knowledge-dir", type=str, default="",
                        help="Path to campaign state dir from training phase "
                             "(loads learned VS, surrogate, technique stats, etc.)")
    args = parser.parse_args()

    # ── Load RMCBench dataset ──
    from evaluation.utils.rmcbench_loader import load_rmcbench
    categories = [args.category] if args.category else None
    levels = [args.level] if args.level else None
    entries = load_rmcbench(
        args.jsonl,
        n=args.num_prompts if args.num_prompts > 0 else None,
        categories=categories,
        levels=levels,
    )
    prompts = [e["prompt"] for e in entries]
    logger.info("Loaded %d prompts from %s", len(prompts), args.jsonl)

    if not prompts:
        logger.error("No prompts found. Check dataset path and filters.")
        sys.exit(1)

    # Print dataset stats
    cat_counts: dict[str, int] = {}
    level_counts: dict[str, int] = {}
    for e in entries:
        cat = e.get("category", "unknown")
        lvl = str(e.get("level", "?"))
        cat_counts[cat] = cat_counts.get(cat, 0) + 1
        level_counts[lvl] = level_counts.get(lvl, 0) + 1
    logger.info("Categories: %s", cat_counts)
    logger.info("Levels: %s", level_counts)

    # ── Victim (OpenRouter) ──
    from adapters.openrouter_victim import OpenRouterVictim
    victim = OpenRouterVictim(
        model=args.victim_model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    logger.info("Victim: %s (OpenRouter)", args.victim_model)

    # ── LLM client (for attacker/refiner/judge) ──
    from llm.llm_client import get_default_client, OpenRouterClient
    if args.attacker_model:
        llm = OpenRouterClient(model=args.attacker_model)
    else:
        llm = get_default_client()
    logger.info("Attacker LLM: %s", llm.model)

    # ── Red Team Agent ──
    from agents.red_team import RedTeamAgent
    red_team = RedTeamAgent(
        llm_client=llm,
        refinement_rounds=3,
    )
    logger.info("Red Team Agent initialized")

    # ── Judge: LLM primary, RuleBased fallback ──
    from evaluation.judges import LLMJudge, RuleBasedJudge
    judge = LLMJudge(llm_client=llm, fallback_judge=RuleBasedJudge())
    logger.info("Judge: LLMJudge + RuleBasedJudge fallback")

    # ── Run evaluation via HarmonyASREvaluator ──
    from evaluation.evaluators import HarmonyASREvaluator
    knowledge_dir = args.knowledge_dir or None
    if knowledge_dir:
        logger.info("Loading campaign knowledge from %s", knowledge_dir)
    evaluator = HarmonyASREvaluator(
        victim=victim,
        judge=judge,
        csv_path="",  # Not used — we pass prompts directly
        red_team_agent=red_team,
        strategist_agent=None,  # No strategist needed for standalone eval
        num_variants=1,
        knowledge_dir=knowledge_dir,
    )

    logger.info("=" * 60)
    logger.info("Starting HARMONY ASR evaluation")
    logger.info("  Prompts:      %d", len(prompts))
    logger.info("  Victim:       %s", args.victim_model)
    logger.info("  Attacker:     %s", llm.model)
    logger.info("  Max retries:  %d", args.max_retries)
    logger.info("  Knowledge:    %s", knowledge_dir or "(none — no learned state)")
    logger.info("=" * 60)

    start = time.time()
    result = evaluator.evaluate(
        prompts=prompts,
        judge=judge,
        max_retries=args.max_retries,
    )
    elapsed = time.time() - start

    # ── Print results ──
    print("\n" + "=" * 60)
    print("HARMONY ASR RESULT (RMCBench)")
    print("=" * 60)
    print(f"  ASR:          {result['asr']:.4f} ({result['successes']}/{result['total']})")
    print(f"  Duration:     {elapsed:.1f}s")
    print(f"  Prompts:      {result['total']}")
    print(f"  Successes:    {result['successes']}")
    print(f"  Failures:     {result['failures']}")
    print(f"  Victim:       {args.victim_model}")
    print(f"  Attacker:     {llm.model}")
    print("=" * 60)

    # ── Per-category breakdown ──
    if entries:
        cat_results: dict[str, dict] = {}
        for i, detail in enumerate(result.get("details", [])):
            if i < len(entries):
                cat = entries[i].get("category", "unknown")
                lvl = entries[i].get("level", 0)
                if cat not in cat_results:
                    cat_results[cat] = {"total": 0, "successes": 0}
                cat_results[cat]["total"] += 1
                if detail["outcome"] == 0:
                    cat_results[cat]["successes"] += 1

        print("\nPer-Category ASR:")
        for cat, stats in sorted(cat_results.items()):
            cat_asr = stats["successes"] / stats["total"] if stats["total"] > 0 else 0
            print(f"  {cat}: {cat_asr:.4f} ({stats['successes']}/{stats['total']})")

    # ── Save detailed JSON ──
    output_path = args.out or str(Path(_project_root) / "outputs" / "rmcbench_asr_result.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Enrich details with dataset metadata
    for i, detail in enumerate(result.get("details", [])):
        if i < len(entries):
            detail["category"] = entries[i].get("category", "unknown")
            detail["level"] = entries[i].get("level", 0)
            detail["pid"] = entries[i].get("pid", i)
            detail["malicious_functionality"] = entries[i].get("malicious functionality", "")

    save_data = {
        **result,
        "duration_s": round(elapsed, 2),
        "victim_model": args.victim_model,
        "attacker_model": llm.model,
        "max_retries": args.max_retries,
        "dataset": args.jsonl,
        "num_prompts": len(prompts),
    }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    logger.info("Results saved to %s", output_path)

    # ── Save human-readable CSV ──
    csv_path = output_path.with_suffix(".csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "pid", "category", "level", "label", "outcome",
                         "technique", "attempts", "goal", "crafted_prompt", "victim_response"])
        for d in result.get("details", []):
            writer.writerow([
                d.get("index", ""),
                d.get("pid", ""),
                d.get("category", ""),
                d.get("level", ""),
                d.get("label", ""),
                d.get("outcome", ""),
                d.get("technique", ""),
                d.get("attempts", 1),
                d.get("goal", ""),
                d.get("crafted_prompt", ""),
                d.get("victim_response", ""),
            ])
    logger.info("Human-readable log saved to %s", csv_path)


if __name__ == "__main__":
    main()
