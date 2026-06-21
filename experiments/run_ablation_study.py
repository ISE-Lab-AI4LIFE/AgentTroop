#!/usr/bin/env python3
"""Multi-seed ablation study runner with variance decomposition & statistical tests.

Usage:
    python experiments/run_ablation_study.py \\
        --configs full_pipeline no_discriminator no_evolution \\
        --num-seeds 10 \\
        --backend openrouter \\
        --num-asr 40 \\
        --num-variants 5

Output: ``outputs/ablation_study/`` with per-seed results CSV + summary JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("ablation_study")
logger.setLevel(logging.INFO)
logging.getLogger("harmony_experiment").setLevel(logging.WARNING)
logging.getLogger("orchestration").setLevel(logging.WARNING)
logging.getLogger("agents").setLevel(logging.WARNING)
logging.getLogger("evaluation").setLevel(logging.WARNING)

RESULTS_DIR = Path(_project_root) / "outputs" / "ablation_study"
ABLATION_CONFIGS_DIR = (
    Path(_project_root) / "experiments" / "configs" / "ablation"
)
BASELINE_CONFIGS_DIR = (
    Path(_project_root) / "experiments" / "configs" / "baseline"
)


def load_ablation_config(config_name: str) -> dict:
    """Load an ablation YAML config by name."""
    # Try ablation configs first, then baseline configs, then direct path
    paths = [
        ABLATION_CONFIGS_DIR / f"{config_name}.yaml",
        BASELINE_CONFIGS_DIR / f"{config_name}.yaml",
        Path(config_name),
    ]
    for p in paths:
        if p.exists():
            with open(p) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(
        f"Ablation config '{config_name}' not found in "
        f"{ABLATION_CONFIGS_DIR}, {BASELINE_CONFIGS_DIR}, or as a path."
    )


def run_single_seed(
    config: dict,
    config_name: str,
    seed_index: int,
    backend: str,
    agentic_backend: str,
    num_asr_str: str,
    num_variants: int,
    max_techniques: int,
    judge_backend: Optional[str] = None,
    judge_model: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one seed of an ablation config and return metrics."""
    from experiments.run_experiment import run_experiment, OUTPUTS_DIR, _run_evaluation, _save_outputs
    from knowledge.episodic import EpisodicMemory
    from knowledge.defense_store import DefenseProgramStore
    from knowledge.causal_graph import CausalGraph
    from knowledge.ontology_memory import OntologyMemory
    from knowledge.scientific_memory import ScientificMemory
    from knowledge.session_memory import SessionMemory
    from knowledge.manager import KnowledgeManager, Target
    from llm.llm_client import get_default_client
    from evaluation.judges import LLMJudge, RuleBasedJudge

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    campaign_id = f"ablation_{config_name}_seed{seed_index}_{timestamp}"
    experiment_id = f"{campaign_id}_run"

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # ── Extract ablation toggles from config ──
    eval_cfg = config.get("evaluation", {})
    prog_disc_enabled = eval_cfg.get("program_discriminator", {}).get("enabled", True)
    tech_selection_mode = eval_cfg.get("technique_selection", {}).get("mode", "ucb")
    num_variants_eval = eval_cfg.get("num_variants", num_variants)
    cognitive_enabled = config.get("cognitive", {}).get("enabled", True)
    strategist_enabled = config.get("strategist", {}).get("enabled", True)
    red_team_enabled = config.get("red_team", {}).get("enabled", True)
    surrogate_enabled = config.get("surrogate", {}).get("enabled", True)
    synthesis_mode = config.get("synthesis", {}).get("mode", "evolutionary")

    redis_url = os.environ.get("HX_REDIS_URL", "redis://localhost:6379/0")
    neo4j_uri = os.environ.get("HX_NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.environ.get("HX_NEO4J_USER", "neo4j")
    neo4j_pass = os.environ.get("HX_NEO4J_PASSWORD", "password")

    episodic = EpisodicMemory(db_path=str(OUTPUTS_DIR / campaign_id / f"{campaign_id}_episodic.db"))
    session = SessionMemory(redis_url=redis_url, ttl=86400)
    defense = DefenseProgramStore(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
    causal = CausalGraph(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
    ontology = OntologyMemory(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)
    scientific = ScientificMemory(uri=neo4j_uri, user=neo4j_user, password=neo4j_pass)

    km = KnowledgeManager(redis_url=redis_url, use_redis=True)
    km.register_store(Target.EPISODIC, episodic)
    km.register_store(Target.DEFENSE_STORE, defense)
    km.register_store(Target.SCIENTIFIC, scientific)
    km.register_store(Target.CAUSAL, causal)
    km.register_store(Target.SESSION, session)

    from prompt_loader import _resolve_path as _resolve_csv
    harmful_csv = os.environ.get(
        "HARMFUL_CSV",
        str(_resolve_csv("")),
    )
    BENIGN_CSV = str(Path(_project_root) / "data" / "benign_prompts.csv")

    # ── Victim ──
    victim_cfg = config["victim"]
    if backend == "openrouter":
        from victim.openrouter import OpenRouterVictim
        from llm.llm_client import OpenRouterClient
        client = OpenRouterClient()
        victim = OpenRouterVictim(
            model_name=victim_cfg["model_name"],
            temperature=victim_cfg["temperature"],
            max_tokens=victim_cfg["max_tokens"],
            client=client,
        )
    elif backend == "ollama":
        from victim.ollama import OllamaVictim
        victim = OllamaVictim(
            ollama_url=victim_cfg["ollama_url"],
            model_name=victim_cfg["model_name"],
            temperature=victim_cfg["temperature"],
            max_tokens=victim_cfg["max_tokens"],
        )
    elif backend == "openai":
        from victim.openai import OpenAIVictim
        from llm.llm_client import OpenAIClient
        client = OpenAIClient()
        victim = OpenAIVictim(
            model_name=victim_cfg["model_name"],
            temperature=victim_cfg["temperature"],
            max_tokens=victim_cfg["max_tokens"],
            client=client,
        )

    llm = get_default_client(backend=agentic_backend)

    if judge_backend or judge_model:
        jb = judge_backend or agentic_backend
        judge_llm = get_default_client(backend=jb)
        if judge_model:
            judge_llm.model = judge_model
    else:
        judge_llm = llm

    # ── Seed ──
    from experiments.run_experiment import seed_episodic_memory as _seed
    num_seeds = config.get("num_seeds", 30)
    full_seeds = config.get("full_seeds", False)
    if full_seeds:
        num_harmful = None
        num_benign = None
    else:
        num_harmful = int(num_seeds * 0.6)
        num_benign = num_seeds - num_harmful

    _, seed_telemetry = _seed(
        episodic, victim, campaign_id, experiment_id,
        harmful_csv, BENIGN_CSV,
        num_harmful=num_harmful, num_benign=num_benign,
    )

    # ── Pre-compute test prompts ──
    from prompt_loader import load_prompts as _load_pt
    test_prompts: set = set()
    try:
        for _seed in (555, 777, 999):
            test_prompts.update(_load_pt(csv_path=harmful_csv, n=50, seed=_seed))
            test_prompts.update(_load_pt(csv_path=BENIGN_CSV, n=50, seed=_seed))
    except Exception:
        pass

    # ── Agents (conditionally created based on toggles) ──
    from agents.cognitive import CognitiveAgent
    from agents.researcher import ResearcherAgent
    from agents.strategist import StrategistAgent
    from agents.red_team import RedTeamAgent
    from synthesis import get_synthesizer

    cognitive_agent = None
    strategist_agent = None
    red_team_agent = None

    if cognitive_enabled:
        cog_cfg = config["cognitive"]
        multi_signal_cfg = (
            cog_cfg.get("multi_signal")
            if cog_cfg.get("adaptive_threshold", False)
            else None
        )
        cognitive_agent = CognitiveAgent(
            episodic_memory=episodic,
            ontology_memory=ontology,
            llm_client=llm,
            llm_backend=agentic_backend,
            anomaly_threshold=cog_cfg["anomaly_threshold"],
            anomaly_selection_config=cog_cfg.get("anomaly_selection"),
            multi_signal_config=multi_signal_cfg,
        )

    if strategist_enabled:
        strat_cfg = config["strategist"]
        tx_cfg = config.get("transforms", {})
        strategist_agent = StrategistAgent(
            episodic_memory=episodic,
            use_llm=strat_cfg["use_llm"],
            llm_client=llm if strat_cfg["use_llm"] else None,
            intervention_budget=config["orchestrator"]["max_interventions"],
            ontology_memory=ontology,
            max_chain_depth=strat_cfg.get("max_chain_depth", 2),
            allowed_transform_names=tx_cfg.get("enabled"),
            blocked_transform_names=tx_cfg.get("disabled"),
            exclude_prompts=test_prompts,
        )

    if red_team_enabled:
        rt_cfg = config.get("red_team", {})
        red_team_agent = RedTeamAgent(
            llm_client=llm,
            llm_backend=agentic_backend,
            refinement_rounds=rt_cfg.get("refinement_rounds", 3),
        )

    res_cfg = config["researcher"]
    if synthesis_mode == "enumeration":
        from synthesis.fitness_guided_synthesizer import FitnessGuidedSynthesizer
        synthesizer = FitnessGuidedSynthesizer(
            max_depth=res_cfg["max_depth"],
            beam_width=res_cfg["beam_width"],
            enumeration_mode=True,
        )
    else:
        synthesizer = get_synthesizer("fitness_guided", config={
            "max_depth": res_cfg["max_depth"],
            "beam_width": res_cfg["beam_width"],
        })

    researcher = ResearcherAgent(
        episodic_memory=episodic,
        defense_store=defense,
        scientific_memory=scientific,
        ontology_memory=ontology,
        synthesizer=synthesizer,
        causal_graph=causal,
    )

    # ── SDE engine (only if cognitive/strategist enabled) ──
    sde_engine = None
    if cognitive_enabled and strategist_enabled:
        from sde.engine import SemanticDiscoveryEngine
        sde_engine = SemanticDiscoveryEngine(
            convergence_std=0.05,
            max_rounds=25,
            use_composite=True,
        )
        victim_name = config.get("victim", {}).get("model_name", "model")
        sde_engine.initialise(victim_name)
        if strategist_agent is not None:
            strategist_agent.sde_engine = sde_engine
            strategist_agent._semantic_enabled = True

        # Patch execute_intervention
        _orig_exec = StrategistAgent.execute_intervention
        def _sde_fed_exec(sself, intervention, victim):
            outcome = _orig_exec(sself, intervention, victim)
            try:
                sde_engine.observe_outcome(
                    prompt=intervention.final_prompt,
                    score=intervention.metadata.get("selection_score", 0.0),
                    outcome=outcome,
                    primitive_name=None,
                )
            except Exception:
                pass
            return outcome
        StrategistAgent.execute_intervention = _sde_fed_exec

        _orig_async_exec = StrategistAgent.async_execute_intervention
        async def _sde_fed_async_exec(sself, intervention, victim):
            outcome = await _orig_async_exec(sself, intervention, victim)
            try:
                sde_engine.observe_outcome(
                    prompt=intervention.final_prompt,
                    score=intervention.metadata.get("selection_score", 0.0),
                    outcome=outcome,
                    primitive_name=None,
                )
            except Exception:
                pass
            return outcome
        StrategistAgent.async_execute_intervention = _sde_fed_async_exec

    # ── Orchestrator ──
    from orchestration.orchestrator import Orchestrator
    orch_cfg = config["orchestrator"]
    orchestrator = Orchestrator(
        cognitive_agent=cognitive_agent,
        strategist_agent=strategist_agent,
        researcher_agent=researcher,
        red_team_agent=red_team_agent,
        knowledge_manager=km,
        session_memory=session,
        victim=victim,
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        max_iterations=orch_cfg["max_iterations"],
        max_interventions=orch_cfg["max_interventions"],
        accuracy_threshold=orch_cfg["accuracy_threshold"],
        error_tolerance=orch_cfg.get("error_tolerance", 0.15),
        synthesis_interval=orch_cfg["synthesis_interval"],
        force_exploration_interval=orch_cfg.get("force_exploration_interval", 3),
        top_k_candidates=res_cfg.get("top_k_candidates", 30),
        seed_telemetry=seed_telemetry,
        holdout_prompts_path=orch_cfg.get("holdout_prompts_path", None),
        convergence_config=orch_cfg.get("convergence"),
        force_exploration_config=orch_cfg.get("force_exploration"),
        min_interventions_per_iteration=orch_cfg.get("min_interventions_per_iteration", 3),
        uncertainty_sampling_fallback=orch_cfg.get("uncertainty_sampling_fallback", True),
        absolute_max_iterations=orch_cfg.get("absolute_max_iterations", 200),
        exclude_prompts=test_prompts,
        surrogate_enabled=surrogate_enabled,
    )

    # ── Run ──
    start = time.time()
    use_async = config.get("use_async", False)
    if use_async:
        import asyncio
        result = asyncio.run(orchestrator.async_run())
    else:
        result = orchestrator.run()
    elapsed = time.time() - start

    # ── Save outputs ──
    campaign_out = Path(_project_root) / "outputs" / "campaign" / campaign_id
    campaign_out.mkdir(parents=True, exist_ok=True)

    # Save campaign state
    from core.jailbreak import _TECHNIQUE_STATS
    from knowledge.campaign_state import save_campaign_state
    campaign_state_dir = save_campaign_state(
        output_dir=str(Path(_project_root) / "outputs" / "campaign"),
        campaign_id=campaign_id,
        version_space=orchestrator.version_space,
        surrogate=getattr(orchestrator, "surrogate", None),
        sde_engine=sde_engine,
        causal_graph=causal,
        technique_stats=_TECHNIQUE_STATS,
    )

    # ── Evaluation ──
    judge = LLMJudge(
        llm_client=judge_llm,
        llm_backend=agentic_backend,
        fallback_judge=RuleBasedJudge(),
    )
    eval_dir = campaign_out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
    }

    # Resolve num_asr
    if num_asr_str == "full":
        from evaluation.utils.test_generator import TestGenerator
        tg = TestGenerator(harmful_csv)
        num_asr = len(tg._prompts)
    else:
        num_asr = int(num_asr_str)
    total_variants = num_asr * num_variants_eval

    # RQ0
    best_id = result.get("best_program_id")
    if best_id:
        try:
            program = None
            record = defense.get(best_id)
            if record is not None:
                program = record.program if hasattr(record, "program") else record
            else:
                vs = result.get("version_space", {})
                prog_dict = vs.get("program_asts", {}).get(best_id)
                if prog_dict:
                    from core.program import Program
                    program = Program.from_dict(prog_dict)
            if program is not None:
                from evaluation.evaluators import RQ0Evaluator
                from prompt_loader import load_prompts
                harmful_test = load_prompts(csv_path=harmful_csv, n=25, seed=555)
                benign_test = load_prompts(csv_path=BENIGN_CSV, n=25, seed=555)
                test_prompts_list = harmful_test + benign_test
                rq0 = RQ0Evaluator(victim=victim, judge=judge)
                rq0_result = rq0.evaluate(program=program, test_prompts=test_prompts_list)
                report["rq0"] = rq0_result
        except Exception as e:
            report["rq0"] = {"accuracy": 0.0, "error": str(e)}

    # ASR (baseline)
    try:
        from evaluation.evaluators import BaselineASREvaluator
        asr_eval = BaselineASREvaluator(
            victim=victim, judge=judge, csv_path=harmful_csv
        )
        asr_result = asr_eval.evaluate(num_prompts=total_variants)
        report["asr"] = asr_result
    except Exception as e:
        report["asr"] = {"asr": 0.0, "error": str(e)}

    # HarmonyX ASR (wired with ablation toggles)
    try:
        from evaluation.evaluators.harmony_x_asr_evaluator import HarmonyXASREvaluator

        harmonyx_asr_eval = HarmonyXASREvaluator(
            victim=victim, judge=judge, csv_path=harmful_csv,
            red_team_agent=red_team_agent,
            num_variants=1,
            knowledge_dir=campaign_state_dir,
            max_techniques=max_techniques,
            program_discriminator_enabled=prog_disc_enabled,
            technique_selection_mode=tech_selection_mode,
        )
        harmonyx_asr_result = harmonyx_asr_eval.evaluate(
            num_prompts=total_variants, num_variants=num_variants_eval,
        )
        report["harmonyx_asr"] = harmonyx_asr_result
    except Exception as e:
        report["harmonyx_asr"] = {"asr": 0.0, "error": str(e)}

    # Save report
    report_path = eval_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    # Extract per-prompt outcomes for McNemar's test
    per_prompt_outcomes = []
    if "details" in report.get("harmonyx_asr", {}):
        for detail in report["harmonyx_asr"]["details"]:
            per_prompt_outcomes.append(detail.get("outcome", 1))

    # Cleanup
    for store in [episodic, defense, causal, ontology, scientific, session]:
        try:
            store.close()
        except Exception:
            pass

    return {
        "config_name": config_name,
        "seed_index": seed_index,
        "campaign_id": campaign_id,
        "elapsed_seconds": round(elapsed, 2),
        "pipeline_success": result.get("success", False),
        "pipeline_best_accuracy": result.get("best_accuracy", 0.0),
        "total_interventions": result.get("total_interventions", 0),
        "total_iterations": result.get("total_iterations", 0),
        "rq0_accuracy": report.get("rq0", {}).get("accuracy", 0.0),
        "baseline_asr": report.get("asr", {}).get("asr", 0.0),
        "harmonyx_asr": report.get("harmonyx_asr", {}).get("asr", 0.0),
        "harmonyx_easr": report.get("harmonyx_asr", {}).get("easr", 0.0),
        "program_blocked": report.get("harmonyx_asr", {}).get("program_blocked", 0),
        "asr_improvement": (
            report.get("harmonyx_asr", {}).get("asr", 0.0)
            - report.get("asr", {}).get("asr", 0.0)
        ),
        "per_prompt_outcomes": per_prompt_outcomes,
        "convergence_iterations": result.get("total_iterations", 0),
        "sample_efficiency": (
            result.get("best_accuracy", 0.0) / max(result.get("total_interventions", 1), 1)
        ),
    }


def main():
    parser = argparse.ArgumentParser(
        description="HARMONY-X multi-seed ablation study",
    )
    parser.add_argument(
        "--configs", nargs="+",
        default=["full_pipeline", "no_program_discriminator"],
        help="Ablation config names (from experiments/configs/ablation/)",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=5,
        help="Number of seeds per config (default: 5)",
    )
    parser.add_argument(
        "--backend", choices=["openrouter", "ollama", "openai"],
        default="openrouter",
    )
    parser.add_argument(
        "--agentic-backend", choices=["openrouter", "openai"],
        default="openai",
    )
    parser.add_argument(
        "--num-asr", type=str, default="40",
        help="Number of prompts for ASR evaluation",
    )
    parser.add_argument(
        "--num-variants", type=int, default=5,
    )
    parser.add_argument(
        "--max-techniques", type=int, default=0,
    )
    parser.add_argument(
        "--judge-backend", choices=["openai", "openrouter"], default=None,
    )
    parser.add_argument(
        "--judge-model", type=str, default=None,
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Path to existing results CSV to resume from",
    )
    parser.add_argument(
        "--seed-start", type=int, default=0,
        help="Starting seed index (for parallel runs)",
    )
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load existing results if resuming
    existing_results: List[Dict[str, Any]] = []
    if args.resume:
        with open(args.resume) as f:
            reader = csv.DictReader(f)
            existing_results = list(reader)
        logger.info("Resumed %d existing results from %s", len(existing_results), args.resume)
        # Determine which (config, seed) pairs are already done
        done_pairs = set()
        for r in existing_results:
            done_pairs.add((r.get("config_name", ""), int(r.get("seed_index", -1))))
    else:
        done_pairs = set()

    all_results: List[Dict[str, Any]] = list(existing_results)

    for config_name in args.configs:
        logger.info("=" * 60)
        logger.info("Config: %s", config_name)
        logger.info("=" * 60)

        try:
            config = load_ablation_config(config_name)
        except FileNotFoundError as e:
            logger.error("Config '%s' not found: %s", config_name, e)
            continue

        for seed_idx in range(args.seed_start, args.seed_start + args.num_seeds):
            key = (config_name, seed_idx)
            if key in done_pairs:
                logger.info("  Skipping seed %d (already done)", seed_idx)
                continue

            logger.info("  Seed %d/%d...", seed_idx + 1, args.seed_start + args.num_seeds)
            try:
                result = run_single_seed(
                    config=config,
                    config_name=config_name,
                    seed_index=seed_idx,
                    backend=args.backend,
                    agentic_backend=args.agentic_backend,
                    num_asr_str=args.num_asr,
                    num_variants=args.num_variants,
                    max_techniques=args.max_techniques,
                    judge_backend=args.judge_backend,
                    judge_model=args.judge_model,
                )
                all_results.append(result)
                logger.info(
                    "    harmonyx_asr=%.4f rq0=%.4f baseline_asr=%.4f "
                    "intv=%d iter=%d time=%.1fs",
                    result["harmonyx_asr"],
                    result["rq0_accuracy"],
                    result["baseline_asr"],
                    result["total_interventions"],
                    result["total_iterations"],
                    result["elapsed_seconds"],
                )
            except Exception as exc:
                logger.error("    Seed %d failed: %s", seed_idx, exc)
                traceback.print_exc()
                all_results.append({
                    "config_name": config_name,
                    "seed_index": seed_idx,
                    "campaign_id": "",
                    "elapsed_seconds": 0.0,
                    "pipeline_success": False,
                    "pipeline_best_accuracy": 0.0,
                    "total_interventions": 0,
                    "total_iterations": 0,
                    "rq0_accuracy": 0.0,
                    "baseline_asr": 0.0,
                    "harmonyx_asr": 0.0,
                    "harmonyx_easr": 0.0,
                    "program_blocked": 0,
                    "asr_improvement": 0.0,
                    "per_prompt_outcomes": [],
                    "convergence_iterations": 0,
                    "sample_efficiency": 0.0,
                    "error": str(exc),
                })

            # Save incremental results after each seed
            csv_path = RESULTS_DIR / f"ablation_results_{timestamp}.csv"
            _write_csv(csv_path, all_results)

    # ── Final analysis ─────────────────────────────────────────────────
    csv_path = RESULTS_DIR / f"ablation_results_{timestamp}.csv"
    _write_csv(csv_path, all_results)

    if len(all_results) > 0:
        summary = _compute_summary(all_results, args)
        summary_path = RESULTS_DIR / f"ablation_summary_{timestamp}.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info("Summary saved to %s", summary_path)

        _print_summary(summary)


def _write_csv(csv_path: Path, results: List[Dict[str, Any]]):
    """Write results to CSV, flattening per-prompt outcomes."""
    fieldnames = [
        "config_name", "seed_index", "campaign_id", "elapsed_seconds",
        "pipeline_success", "pipeline_best_accuracy",
        "total_interventions", "total_iterations",
        "rq0_accuracy", "baseline_asr",
        "harmonyx_asr", "harmonyx_easr",
        "program_blocked", "asr_improvement",
        "convergence_iterations", "sample_efficiency",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in fieldnames}
            writer.writerow(row)


def _compute_summary(
    results: List[Dict[str, Any]],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Compute per-config summary statistics with bootstrap CIs."""
    from evaluation.metrics.statistical_tests import (
        bootstrap_ci,
        mcnemar_test,
        variance_decomposition,
        cohens_h,
        build_ablation_summary,
    )

    config_names = sorted(set(r["config_name"] for r in results))

    # Group results by config
    grouped: Dict[str, List[Dict[str, Any]]] = {
        cfg: [r for r in results if r["config_name"] == cfg]
        for cfg in config_names
    }

    summary: Dict[str, Any] = {
        "study_metadata": {
            "timestamp": datetime.now().isoformat(),
            "num_configs": len(config_names),
            "num_seeds": args.num_seeds,
            "backend": args.backend,
            "agentic_backend": args.agentic_backend,
            "num_asr": args.num_asr,
            "num_variants": args.num_variants,
            "configs": config_names,
        },
        "per_config": {},
        "pairwise_comparisons": {},
        "variance_decomposition": {},
    }

    # Per-config bootstrap CIs
    metrics = [
        "harmonyx_asr", "rq0_accuracy", "baseline_asr",
        "total_interventions", "total_iterations",
        "program_blocked", "asr_improvement",
        "sample_efficiency",
    ]

    for cfg in config_names:
        summary["per_config"][cfg] = {}
        for metric in metrics:
            values = [
                r.get(metric, 0.0)
                for r in grouped[cfg]
                if r.get("pipeline_success", False) or metric in ("baseline_asr",)
            ]
            if not values:
                summary["per_config"][cfg][metric] = {
                    "statistic": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                    "std_error": 0.0, "n": 0,
                }
                continue
            ci = bootstrap_ci(values, statistic="mean", random_seed=42)
            ci["n"] = len(values)
            ci["values"] = [round(v, 4) for v in values]
            summary["per_config"][cfg][metric] = ci

        # Successful runs count
        n_success = sum(1 for r in grouped[cfg] if r.get("pipeline_success", False))
        summary["per_config"][cfg]["n_success"] = n_success
        summary["per_config"][cfg]["n_total"] = len(grouped[cfg])

    # Pairwise comparisons (McNemar for ASR)
    for i, cfg_a in enumerate(config_names):
        for cfg_b in config_names[i + 1:]:
            key = f"{cfg_a}_vs_{cfg_b}"

            # Get per-prompt outcomes from first successful seed of each config
            outcomes_a = None
            outcomes_b = None
            for r in grouped[cfg_a]:
                if r.get("per_prompt_outcomes"):
                    outcomes_a = r["per_prompt_outcomes"]
                    break
            for r in grouped[cfg_b]:
                if r.get("per_prompt_outcomes"):
                    outcomes_b = r["per_prompt_outcomes"]
                    break

            if outcomes_a and outcomes_b and len(outcomes_a) == len(outcomes_b):
                mcnemar = mcnemar_test(outcomes_a, outcomes_b)
                # Effect size
                asr_a = sum(1 for o in outcomes_a if o == 0) / len(outcomes_a)
                asr_b = sum(1 for o in outcomes_b if o == 0) / len(outcomes_b)
                h = cohens_h(asr_a, asr_b)
                summary["pairwise_comparisons"][key] = {
                    "mcnemar": mcnemar,
                    "cohens_h": h,
                    "asr_a": round(asr_a, 4),
                    "asr_b": round(asr_b, 4),
                }

    # Variance decomposition
    for metric in ["harmonyx_asr", "rq0_accuracy", "total_interventions"]:
        metric_groups = {
            cfg: [
                r.get(metric, 0.0)
                for r in grouped[cfg]
                if r.get("pipeline_success", False)
            ]
            for cfg in config_names
        }
        if all(len(v) > 0 for v in metric_groups.values()):
            summary["variance_decomposition"][metric] = variance_decomposition(metric_groups)

    return summary


def _print_summary(summary: Dict[str, Any]) -> None:
    """Print a formatted markdown summary table to stdout."""
    configs = summary["study_metadata"]["configs"]
    metrics = [
        ("harmonyx_asr", "HarmonyX ASR"),
        ("baseline_asr", "Baseline ASR"),
        ("asr_improvement", "ASR Improvement"),
        ("rq0_accuracy", "Program Accuracy"),
        ("total_interventions", "Interventions"),
        ("total_iterations", "Iterations"),
        ("program_blocked", "Blocked"),
        ("sample_efficiency", "Sample Efficiency"),
    ]

    print()
    print("=" * 100)
    print("ABLATION STUDY RESULTS")
    print("=" * 100)
    print(f"Configs: {', '.join(configs)}")
    print(f"Seeds per config: {summary['study_metadata']['num_seeds']}")
    print(f"Backend: {summary['study_metadata']['backend']}")
    print()

    for metric_key, metric_label in metrics:
        print(f"**{metric_label}**")
        header = f"| {'Config':<30} | {'Mean':<10} | {'CI Lower':<10} | {'CI Upper':<10} | {'Std Err':<10} | N |"
        sep = "|" + "-" * 32 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 3 + "|"
        print(header)
        print(sep)
        for cfg in configs:
            ci = summary["per_config"].get(cfg, {}).get(metric_key, {})
            mean = ci.get("statistic", 0.0)
            lo = ci.get("ci_lower", 0.0)
            hi = ci.get("ci_upper", 0.0)
            se = ci.get("std_error", 0.0)
            n = ci.get("n", 0)
            print(f"| {cfg:<30} | {mean:<10.4f} | {lo:<10.4f} | {hi:<10.4f} | {se:<10.4f} | {n:<3} |")
        print()

    # Pairwise comparisons
    if summary["pairwise_comparisons"]:
        print("**Pairwise Comparisons (McNemar's Test)**")
        for key, comp in summary["pairwise_comparisons"].items():
            m = comp["mcnemar"]
            h = comp["cohens_h"]
            sig = "**SIGNIFICANT**" if m["significant"] else "n.s."
            print(f"  {key}: χ²={m['chi2']}, p={m['p_value']}, {sig}")
            print(f"    ASR: A={comp['asr_a']:.4f} vs B={comp['asr_b']:.4f}")
            print(f"    Cohen's h={h['h']:.4f} ({h['interpretation']})")
            print(f"    Discordant pairs: A=0,B=1: {m['n_01']}, A=1,B=0: {m['n_10']}")
            print(f"    {m['interpretation']}")
            print()

    # Variance decomposition
    if summary["variance_decomposition"]:
        print("**Variance Decomposition (ICC)**")
        for metric, vd in summary["variance_decomposition"].items():
            print(f"  {metric}: ICC={vd['icc']:.4f} ({vd['interpretation']})")
            print(f"    Total var: {vd['total_variance']:.6f}")
            print(f"    Within-config: {vd['within_config_variance']:.6f}")
            print(f"    Between-config: {vd['between_config_variance']:.6f}")

    print("=" * 100)


if __name__ == "__main__":
    main()
