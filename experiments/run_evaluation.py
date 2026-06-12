#!/usr/bin/env python3
"""Run all HARMONY-X evaluations (RQ0–RQ3, ASR) on a completed campaign.

Usage:
    cd /path/to/HARMONY_X
    python experiments/run_evaluation.py --campaign llama31_8b_test_001 \
        --program-id dfp_91f11d6edc31 \
        --output-dir evaluation/reports
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("run_evaluation")

from adapters.base_victim import BaseVictim
from core.program import Program

from knowledge.defense_store import DefenseProgramStore
from knowledge.episodic import EpisodicMemory
from knowledge.scientific_memory import ScientificMemory

from evaluation.judges import Judge, RuleBasedJudge, LLMJudge
from evaluation.evaluators import (
    RQ0Evaluator,
    RQ1Evaluator,
    RQ2Evaluator,
    RQ3Evaluator,
    BaselineASREvaluator,
)
from evaluation.utils import VictimWrapper, TestGenerator


def load_program(
    defense_store: DefenseProgramStore,
    program_id: str,
) -> Optional[Program]:
    record = defense_store.get(program_id)
    if record is None:
        return None
    if hasattr(record, "program"):
        return record.program
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HARMONY-X evaluation: RQ0–RQ3 + ASR",
    )
    parser.add_argument("--campaign", required=True, help="Campaign ID to evaluate")
    parser.add_argument("--experiment", default=None, help="Experiment ID (optional)")
    parser.add_argument("--program-id", default=None, help="Program ID to evaluate (RQ0)")
    parser.add_argument("--victim", default=None, help="Victim module path (e.g. adapters.toy_victims.rule_based.KeywordFilterVictim)")
    parser.add_argument("--judge", choices=["rule", "llm"], default="llm", help="Judge type")
    parser.add_argument("--llm-model", default="gemma-4-31b-it", help="LLM model for LLMJudge")
    parser.add_argument("--baseline-campaign", default=None, help="Baseline campaign ID for RQ1 random probing comparison")
    parser.add_argument("--baseline-experiment", default=None, help="Baseline experiment ID for RQ1 random probing comparison")
    parser.add_argument("--num-test-prompts", type=int, default=50, help="Number of test prompts (RQ0, ASR)")
    parser.add_argument("--accuracy-threshold", type=float, default=0.85, help="Accuracy threshold (RQ0, RQ1)")
    parser.add_argument("--transfer-threshold", type=float, default=0.9, help="Transfer accuracy threshold (RQ3)")
    parser.add_argument(
        "--prior-campaign",
        default=None,
        help="Prior campaign ID for transfer speed comparison (no-transfer baseline, RQ3)",
    )
    parser.add_argument("--annotation-file", default=None, help="Annotation JSON file (RQ2)")
    parser.add_argument("--prompt-csv", default="", help="Path to prompt.csv (RMCBench harmful prompts)")
    parser.add_argument("--output-dir", default="evaluation/reports", help="Output directory")
    parser.add_argument("--episodic-db", default=None, help="Path to episodic DB (default: <campaign>_episodic.db)")
    parser.add_argument("--neo4j-uri", default="bolt://localhost:7687")
    parser.add_argument("--neo4j-user", default="neo4j")
    parser.add_argument("--neo4j-password", default="password")

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report: dict[str, Any] = {
        "campaign_id": args.campaign,
        "experiment_id": args.experiment,
        "timestamp": timestamp,
        "results": {},
    }

    # Judge
    judge: Judge
    if args.judge == "llm":
        judge = LLMJudge(model_name=args.llm_model, fallback_judge=RuleBasedJudge())
        logger.info("Using LLMJudge (model=%s)", args.llm_model)
    else:
        judge = RuleBasedJudge()
        logger.info("Using RuleBasedJudge")

    # Victim
    victim: Optional[BaseVictim] = None
    if args.victim:
        import importlib
        module_path, cls_name = args.victim.rsplit(".", 1)
        mod = importlib.import_module(module_path)
        victim_cls = getattr(mod, cls_name)
        victim = victim_cls()
        logger.info("Victim: %s", args.victim)

    # Knowledge stores
    db_path = args.episodic_db or f"{args.campaign}_episodic.db"
    episodic = EpisodicMemory(db_path=db_path)
    defense = DefenseProgramStore(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )
    scientific = ScientificMemory(
        uri=args.neo4j_uri,
        user=args.neo4j_user,
        password=args.neo4j_password,
    )

    # RQ0: Program accuracy
    if args.program_id and victim is not None:
        logger.info("=" * 60)
        logger.info("RQ0: Program Accuracy")
        logger.info("=" * 60)
        program = load_program(defense, args.program_id)
        if program is not None:
            rq0 = RQ0Evaluator(victim=victim, judge=judge, csv_path=args.prompt_csv)
            result = rq0.evaluate(
                program=program,
                num_test_prompts=args.num_test_prompts,
            )
            report["results"]["RQ0"] = result
            logger.info("RQ0 result: %s", json.dumps(result, indent=2))
        else:
            logger.warning("Program %s not found in DefenseProgramStore", args.program_id)

    # RQ1: Intervention efficiency
    logger.info("=" * 60)
    logger.info("RQ1: Intervention Efficiency")
    logger.info("=" * 60)
    rq1 = RQ1Evaluator(episodic)
    result = rq1.evaluate(
        campaign_id=args.campaign,
        experiment_id=args.experiment,
        threshold=args.accuracy_threshold,
        baseline_campaign_id=args.baseline_campaign,
        baseline_experiment_id=args.baseline_experiment,
    )
    report["results"]["RQ1"] = result
    logger.info("RQ1 result: %s", json.dumps(result, indent=2))

    # RQ2: Explanation score (if annotation file provided)
    if args.annotation_file and os.path.exists(args.annotation_file):
        logger.info("=" * 60)
        logger.info("RQ2: Explanation Score")
        logger.info("=" * 60)
        rq2 = RQ2Evaluator()
        result = rq2.evaluate(
            annotation_path=args.annotation_file,
            program_id=args.program_id or "",
        )
        report["results"]["RQ2"] = result
        logger.info("RQ2 result: %s", json.dumps(result, indent=2))

    # RQ3: Transfer speed (if prior campaign provided)
    if args.prior_campaign:
        logger.info("=" * 60)
        logger.info("RQ3: Transfer Speed")
        logger.info("=" * 60)
        rq3 = RQ3Evaluator(episodic)
        result = rq3.evaluate(
            prior_campaign_id=args.prior_campaign,
            target_campaign_id=args.campaign,
            prior_experiment_id=args.experiment,
            target_experiment_id=args.experiment,
            threshold=args.transfer_threshold,
        )
        report["results"]["RQ3"] = result
        logger.info("RQ3 result: %s", json.dumps(result, indent=2))

    # ASR
    if victim is not None:
        logger.info("=" * 60)
        logger.info("ASR: Attack Success Rate")
        logger.info("=" * 60)
        asr_eval = BaselineASREvaluator(victim=victim, judge=judge, csv_path=args.prompt_csv)
        result = asr_eval.evaluate(num_prompts=args.num_test_prompts)
        report["results"]["ASR"] = result
        logger.info("ASR result: %s", json.dumps(result, indent=2))

    # Save report
    report_path = output_dir / f"evaluation_report_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    # Cleanup
    episodic.close()
    defense.close()
    scientific.close()

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
