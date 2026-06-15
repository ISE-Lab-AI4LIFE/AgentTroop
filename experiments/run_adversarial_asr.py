#!/usr/bin/env python3
"""Standalone adversarial ASR evaluation — craft prompts with the learned program.

Usage:
    python run_adversarial_asr.py --config config_validate.yaml
    python run_adversarial_asr.py --program path/to/program.json --num-prompts 50
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

EXP_DIR = Path(__file__).resolve().parent
OUTPUTS_DIR = Path(EXP_DIR).parent / "outputs" / "campaign"
LOGS_DIR = EXP_DIR / "logs"

_project_root = str(EXP_DIR.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if str(EXP_DIR) not in sys.path:
    sys.path.insert(0, str(EXP_DIR))

logger = logging.getLogger("adversarial_asr")


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
    for lib in ("neo4j", "urllib3", "httpx", "faiss"):
        logging.getLogger(lib).setLevel(logging.WARNING)


def load_config(path: str) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def load_program(path: str) -> Any:
    from core.program import Program
    with open(path) as f:
        data = json.load(f)
    prog_data = data.get("program") or data
    return Program.from_dict(prog_data)


def load_harmful_csv() -> str:
    from prompt_loader import _resolve_path as _resolve_csv
    return str(_resolve_csv(os.environ.get("HARMFUL_CSV", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Adversarial ASR evaluation")
    parser.add_argument(
        "--config", default=str(EXP_DIR / "config.yaml"),
        help="Experiment config YAML (default: config.yaml)",
    )
    parser.add_argument(
        "--program", default=str(OUTPUTS_DIR / "final_program.json"),
        help="Path to saved program JSON (default: outputs/final_program.json)",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=30,
        help="Number of jailbreak test prompts (default: 30)",
    )
    parser.add_argument(
        "--max-depth", type=int, default=2,
        help="Max transform chain depth (default: 2)",
    )
    parser.add_argument(
        "--csv", type=str, default="",
        help="Override harmful prompts CSV path",
    )
    args = parser.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(LOGS_DIR / f"adversarial_asr_{timestamp}.log")
    setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Adversarial ASR — standalone evaluation")
    logger.info("=" * 60)
    logger.info("Config:  %s", args.config)
    logger.info("Program: %s", args.program)
    logger.info("Log:     %s", os.path.abspath(log_file))

    # Load config
    config = load_config(args.config)
    victim_cfg = config["victim"]

    # Load program
    prog_path = Path(args.program)
    if not prog_path.exists():
        logger.error("Program file not found: %s", prog_path)
        sys.exit(1)
    program = load_program(str(prog_path))
    logger.info("Loaded program: %s", program.id)

    # Create victim
    from victim.ollama import OllamaVictim
    victim = OllamaVictim(
        ollama_url=victim_cfg["ollama_url"],
        model_name=victim_cfg["model_name"],
        temperature=victim_cfg["temperature"],
        max_tokens=victim_cfg["max_tokens"],
    )
    logger.info("Victim: %s (%s)", victim.model_name, victim.ollama_url)

    # Create judge
    from llm.llm_client import get_default_client
    from evaluation.judges import LLMJudge, RuleBasedJudge
    llm = get_default_client()
    judge = LLMJudge(llm_client=llm, fallback_judge=RuleBasedJudge())

    # Resolve CSV path for test prompts
    harmful_csv = args.csv or load_harmful_csv()
    logger.info("Harmful CSV: %s", harmful_csv)

    # Run evaluation
    from evaluation.evaluators import HarmonyXASREvaluator
    evaluator = HarmonyXASREvaluator(
        victim=victim, judge=judge, csv_path=harmful_csv,
        red_team_agent=red_team, num_variants=1,
    )
    try:
        from knowledge.campaign_state import load_campaign_state
        from pathlib import Path
        campaign_out = OUTPUTS_DIR / campaign_id
        if campaign_out.exists():
            evaluator._knowledge_dir = str(campaign_out)
            evaluator._load_knowledge()
    except Exception:
        pass
    result = evaluator.evaluate(num_prompts=args.num_prompts)

    # Print results
    logger.info("")
    logger.info("=" * 60)
    logger.info("  HARMONY_X ASR Result")
    logger.info("=" * 60)
    logger.info("  ASR:                %.4f (%d/%d)",
                result["asr"],
                result["successes"],
                result["total"])
    logger.info("  EASR:               %.4f (%d/%d)",
                result["easr"],
                result["successes"],
                result["attempted"])
    logger.info("  Program blocked:    %d", result.get("program_blocked", 0))
    logger.info("=" * 60)

    # Save result
    eval_dir = OUTPUTS_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report_path = eval_dir / f"adversarial_asr_{timestamp}.json"
    with open(report_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Report saved to %s", report_path)

    sys.exit(0)


if __name__ == "__main__":
    main()
