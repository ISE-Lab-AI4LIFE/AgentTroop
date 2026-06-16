#!/usr/bin/env python3
"""HARMONY-X experiment: reverse-engineer LLM safety layer."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Ensure project root is on sys.path (NOT experiment dir)
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# ── Paths ──
EXP_DIR = Path(__file__).resolve().parent
LOGS_DIR = EXP_DIR / "logs"
OUTPUTS_DIR = Path(EXP_DIR).parent / "outputs" / "campaign"
CONFIG_PATH = EXP_DIR / "configs" / "experiment_config.yaml"
BENIGN_CSV = str(Path(EXP_DIR).parent / "data" / "benign_prompts.csv")

logger = logging.getLogger("harmony_experiment")


# ── Helpers ─────────────────────────────────────────────────────────────────


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


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


def _load_benign_prompts(path: str, n: Optional[int] = 80) -> List[str]:
    """Load benign prompts from a CSV file with a ``prompt`` column.

    Args:
        path: Path to CSV file.
        n: Number of prompts to return (``None`` = all).
    """
    import csv
    prompts: List[str] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return prompts
        # Case-insensitive column lookup
        prompt_col = "prompt"
        for col in reader.fieldnames:
            if col.strip().lower() == "prompt":
                prompt_col = col
                break
        for row in reader:
            val = row.get(prompt_col, "").strip()
            if val:
                prompts.append(val)
            if n is not None and len(prompts) >= n:
                break
    return prompts


def _check_victim_compliance(victim: Any, harmful_csv: str, benign_csv: str) -> bool:
    """Send 5 harmful + 5 benign prompts to victim and verify it produces
    differing outcomes.  Returns True if at least 1 harmful REFUSE and
    1 benign ACCEPT were observed."""
    from prompt_loader import load_prompts

    harmful = load_prompts(csv_path=harmful_csv, n=5, seed=99)
    benign = _load_benign_prompts(benign_csv, n=5)

    logger.info("─" * 50)
    logger.info("Victim compliance check (5 harmful + 5 benign prompts)")
    logger.info("─" * 50)

    harmful_refuse = 0
    benign_accept = 0

    for p in harmful:
        outcome = victim.respond(p)
        label = "REFUSE" if outcome else "ACCEPT"
        if outcome:
            harmful_refuse += 1
        logger.info("  harm   → %s | %s", label, p[:60])

    for p in benign:
        outcome = victim.respond(p)
        label = "REFUSE" if outcome else "ACCEPT"
        if not outcome:
            benign_accept += 1
        logger.info("  benign → %s | %s", label, p[:60])

    logger.info("─" * 50)
    logger.info("harmful REFUSE=%d/5  benign ACCEPT=%d/5", harmful_refuse, benign_accept)

    if harmful_refuse == 0 or benign_accept == 0:
        logger.warning(
            "Victim compliance check FAILED: harmful_refuse=%d, benign_accept=%d. "
            "Pipeline may not learn meaningful patterns.",
            harmful_refuse, benign_accept,
        )
        return False
    return True


def seed_episodic_memory(
    episodic_memory: Any,
    victim: Any,
    campaign_id: str,
    experiment_id: str,
    harmful_csv: str,
    benign_csv: str,
    num_harmful: int = 120,
    num_benign: int = 80,
) -> Tuple[int, Dict[str, Any]]:
    """Seed episodic memory with diverse prompt variants across three
    transformation tiers.

    Uses :class:`MultiTierSeedStrategy` to generate semantically meaningful
    anomalies — not just encoding-robustness artifacts.  See
    ``seed_strategy.py`` for the full tier framework.

    Returns
    -------
    (count, telemetry)
        Number of episodes seeded and a telemetry dict with
        ``variant_count_by_family``, ``variant_rate_by_family``, etc.
    """
    from knowledge.episodic import EpisodicMemory, Episode, InterventionRecord
    from prompt_loader import load_prompts
    from seed_strategy import MultiTierSeedStrategy

    harmful = load_prompts(csv_path=harmful_csv, n=num_harmful)
    benign = _load_benign_prompts(benign_csv, n=num_benign)

    strategy = MultiTierSeedStrategy(
        variants_per_prompt={
            "tier1_semantic": 8,
            "tier2_structural": 4,
            "tier3_encoding": 3,
        },
        tier3_max_ratio=0.30,
        seed=42,
    )

    now = time.time()
    count = 0
    all_variants: List[Tuple[str, str, str, str, Dict, int]] = []

    for base_prompt in harmful:
        variants = strategy.generate_variants(base_prompt, tag="harmful")
        for v in variants:
            all_variants.append((
                base_prompt, v["tag"],
                v["transform_meta"]["name"],
                v["final"],
                v["transform_meta"],
                victim.respond(v["final"]),
            ))

    for base_prompt in benign:
        variants = strategy.generate_variants(base_prompt, tag="benign")
        for v in variants:
            all_variants.append((
                base_prompt, v["tag"],
                v["transform_meta"]["name"],
                v["final"],
                v["transform_meta"],
                victim.respond(v["final"]),
            ))

    for idx, (base, tag, tx_name, final, meta, outcome) in enumerate(all_variants):
        ep = Episode(
            episode_id=f"seed_{idx}",
            intervention=InterventionRecord(
                intervention_id=f"seed_{idx}",
                prompt=base,
                transforms=[meta] if meta.get("name") else [],
                final_prompt=final,
                strategy_name="seed",
                agent_name="run_experiment",
                timestamp=now + idx * 0.001,
            ),
            victim_name=victim.name,
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            outcome=outcome,
            raw_response=str(outcome),
            created_at=now + idx * 0.001,
        )
        episodic_memory.save_episode(ep)
        count += 1

    telemetry = strategy.telemetry_report()
    total = telemetry.get("total_variants", 0)
    t1 = telemetry["variant_count_by_family"].get("tier1_semantic", 0)
    t2 = telemetry["variant_count_by_family"].get("tier2_structural", 0)
    t3 = telemetry["variant_count_by_family"].get("tier3_encoding", 0)
    t3_ratio = (t3 / total) if total else 0
    t3_status = "OK" if (t3_ratio <= 0.30) else "EXCEEDS 30% LIMIT"
    logger.info(
        "Seeded %d baseline episodes (%d harmful + %d benign prompts) "
        "T1=%d T2=%d T3=%d "
        "T3 ratio=%.1f%% (%s)",
        count, len(harmful), len(benign),
        t1, t2, t3,
        t3_ratio * 100,
        t3_status,
    )
    return count, telemetry


# ── Main experiment ─────────────────────────────────────────────────────────

def run_experiment(config: dict, backend: str = "ollama",
                   campaign_prefix: Optional[str] = None,
                   prior_campaign_id: Optional[str] = None,
                   agentic_backend: str = "openai",
                   num_asr_str: str = "40",
                   num_variants: int = 5) -> Dict[str, Any]:
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    cfg = config["orchestrator"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if campaign_prefix is None:
        model_name = config.get("victim", {}).get("model_name", "model")
        campaign_prefix = re.sub(r'[^a-zA-Z0-9_]+', '_', model_name).strip('_')
    campaign_id = f"{campaign_prefix}_{timestamp}"
    experiment_id = f"{campaign_id}_run"

    # ── Create stores ──
    from knowledge.episodic import EpisodicMemory
    from knowledge.defense_store import DefenseProgramStore
    from knowledge.causal_graph import CausalGraph
    from knowledge.ontology_memory import OntologyMemory
    from knowledge.scientific_memory import ScientificMemory
    from knowledge.session_memory import SessionMemory
    from knowledge.manager import KnowledgeManager, Target

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

    # Sync primitives
    from core.primitive import default_registry
    ontology.sync_to_registry(overwrite=True)
    logger.info("Synced %d primitives to ontology", len(default_registry.list_primitives()))

    # KnowledgeManager
    km = KnowledgeManager(redis_url=redis_url, use_redis=True)
    km.register_store(Target.EPISODIC, episodic)
    km.register_store(Target.DEFENSE_STORE, defense)
    km.register_store(Target.SCIENTIFIC, scientific)
    km.register_store(Target.CAUSAL, causal)
    km.register_store(Target.SESSION, session)

    # ── CSV dataset paths ──
    from prompt_loader import _resolve_path as _resolve_csv
    harmful_csv = os.environ.get(
        "HARMFUL_CSV",
        str(_resolve_csv("")),
    )
    logger.info("Harmful CSV: %s", harmful_csv)
    logger.info("Benign CSV:  %s", BENIGN_CSV)

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
    logger.info("Victim: %s (via %s)", victim.model_name, backend)
    logger.info("Victim metadata: %s", victim.get_metadata())

    # ── Compliance check ──
    compliant = _check_victim_compliance(victim, harmful_csv, BENIGN_CSV)
    if not compliant:
        logger.warning(
            "Victim shows little behavioural variation — "
            "continuing anyway (set HX_FORCE=1 to skip this warning)"
        )

    # ── Seed ──
    num_seeds = config.get("num_seeds", 30)
    full_seeds = config.get("full_seeds", False)

    if full_seeds:
        num_harmful = None
        num_benign = None
    else:
        num_harmful = int(num_seeds * 0.6)
        num_benign = num_seeds - num_harmful

    _, seed_telemetry = seed_episodic_memory(
        episodic, victim, campaign_id, experiment_id,
        harmful_csv, BENIGN_CSV,
        num_harmful=num_harmful, num_benign=num_benign,
    )
    logger.info("Seed telemetry: %s", json.dumps(seed_telemetry, indent=2))

    # ── Agents ──
    from agents.cognitive import CognitiveAgent
    from agents.researcher import ResearcherAgent
    from agents.strategist import StrategistAgent
    from agents.red_team import RedTeamAgent
    from llm.llm_client import get_default_client
    from synthesis import get_synthesizer

    llm = get_default_client(backend=agentic_backend)

    cog_cfg = config["cognitive"]
    seed_gen_cfg = config.get("seed_generation", {})
    adaptive_exp = seed_gen_cfg.get("adaptive_expansion", False)
    multi_signal_cfg = cog_cfg.get("multi_signal") if cog_cfg.get("adaptive_threshold", False) else None

    cognitive = CognitiveAgent(
        episodic_memory=episodic,
        ontology_memory=ontology,
        llm_client=llm,
        llm_backend=agentic_backend,
        anomaly_threshold=cog_cfg["anomaly_threshold"],
        anomaly_selection_config=cog_cfg.get("anomaly_selection"),
        multi_signal_config=multi_signal_cfg,
    )

    # ── Adaptive expansion (forced variation when no anomalies) ──
    if adaptive_exp:
        from seed_strategy import MultiTierSeedStrategy
        from knowledge.episodic import InterventionRecord, Episode, EpisodeFilter

        max_expansion_rounds = config.get("seed_generation", {}).get(
            "max_expansion_rounds", 5
        )
        exp_factor = config.get("seed_generation", {}).get(
            "expansion_factor", 2
        )
        # Count current seed variants per base prompt
        all_eps = episodic.filter_episodes(
            EpisodeFilter(campaign_id=campaign_id, experiment_id=experiment_id)
        )
        eps_by_prompt: Dict[str, List[Any]] = {}
        for ep in all_eps:
            if hasattr(ep, 'intervention') and ep.intervention.prompt:
                eps_by_prompt.setdefault(ep.intervention.prompt, []).append(ep)

        base_variants_per_prompt = max(
            (len(v) for v in eps_by_prompt.values()), default=15
        )

        exp_strategy = MultiTierSeedStrategy(seed=99)
        expanded_any = False
        for round_idx in range(max_expansion_rounds):
            # Detect anomalies from current episodes
            anoms = cognitive.detect_anomalies(
                campaign_id=campaign_id, experiment_id=experiment_id
            )
            if anoms:
                n_pairwise = sum(1 for a in anoms if a.anomaly_source not in ("embedding", "length"))
                logger.info(
                    "Adaptive expansion round %d/%d: found %d anomalies "
                    "(pairwise=%d, multi-signal=%d) — stopping expansion",
                    round_idx + 1, max_expansion_rounds,
                    len(anoms), n_pairwise, len(anoms) - n_pairwise,
                )
                break

            if not eps_by_prompt:
                logger.info("No episodes to expand — stopping expansion")
                break

            # Expand: double variants per prompt
            target = base_variants_per_prompt * (exp_factor ** (round_idx + 1))
            new_eps: List[Episode] = []
            now = time.time()
            idx_offset = len(all_eps)

            for base_prompt, existing in eps_by_prompt.items():
                existing_count = len(existing)
                extra = exp_strategy.generate_additional_variants(
                    base_prompt=base_prompt,
                    tag="harmful" if "harm" in base_prompt.lower() else "benign",
                    existing_count=existing_count,
                    target_count=target,
                )
                if not extra:
                    continue

                # Tag source prompt for grouping
                source_tag = "harmful"
                for ep in existing:
                    tx = ep.intervention.transforms or [] if hasattr(ep, 'intervention') else []
                    source_tag = tx[0].get("source_tag", "harmful") if tx and isinstance(tx[0], dict) else "harmful"
                    break

                for j, v in enumerate(extra):
                    outcome = victim.respond(v["final"])
                    new_ep = Episode(
                        episode_id=f"seed_expand_{round_idx}_{idx_offset + j}",
                        intervention=InterventionRecord(
                            intervention_id=f"seed_expand_{round_idx}_{idx_offset + j}",
                            prompt=base_prompt,
                            transforms=[v["transform_meta"]] if v["transform_meta"].get("name") else [],
                            final_prompt=v["final"],
                            strategy_name="seed_expand",
                            agent_name="adaptive_expansion",
                            timestamp=now + j * 0.001,
                        ),
                        victim_name=victim.name,
                        campaign_id=campaign_id,
                        experiment_id=experiment_id,
                        outcome=outcome,
                    )
                    new_eps.append(new_ep)

            if not new_eps:
                logger.info(
                    "Adaptive expansion round %d: no new variants generated — "
                    "exhausted transform pool", round_idx + 1,
                )
                break

            # Store new episodes in episodic memory
            for ep in new_eps:
                episodic.save_episode(ep)
                all_eps.append(ep)
                eps_by_prompt.setdefault(ep.intervention.prompt, []).append(ep)

            expanded_any = True
            logger.info(
                "Adaptive expansion round %d/%d: added %d episodes "
                "(target=%d variants/group, total=%d)",
                round_idx + 1, max_expansion_rounds,
                len(new_eps), target, len(all_eps),
            )

        if not expanded_any:
            logger.info(
                "No adaptive expansion needed (%d anomalies from initial seed)",
                len(cognitive.detect_anomalies(
                    campaign_id=campaign_id, experiment_id=experiment_id
                )),
            )

    # ── Pre-compute test prompts (prevent data leakage) ──
    from prompt_loader import load_prompts as _load_pt
    test_prompts: set = set()
    try:
        for _seed in (555, 777, 999):
            test_prompts.update(_load_pt(csv_path=harmful_csv, n=50, seed=_seed))
            test_prompts.update(_load_pt(csv_path=BENIGN_CSV, n=50, seed=_seed))
        logger.info("Pre-computed %d test prompts for exclusion (leakage prevention)", len(test_prompts))
    except Exception as exc:
        logger.warning("Failed to pre-compute test prompts: %s", exc)

    strat_cfg = config["strategist"]
    tx_cfg = config.get("transforms", {})
    strategist = StrategistAgent(
        episodic_memory=episodic,
        use_llm=strat_cfg["use_llm"],
        llm_client=llm if strat_cfg["use_llm"] else None,
        intervention_budget=cfg["max_interventions"],
        ontology_memory=ontology,
        max_chain_depth=strat_cfg.get("max_chain_depth", 2),
        allowed_transform_names=tx_cfg.get("enabled"),
        blocked_transform_names=tx_cfg.get("disabled"),
        exclude_prompts=test_prompts,
    )

    # ── Red Team Agent (LLM prompt refiner — pure refinement, no technique) ──
    rt_cfg = config.get("red_team", {})
    red_team = RedTeamAgent(
        llm_client=llm,
        llm_backend=agentic_backend,
        refinement_rounds=rt_cfg.get("refinement_rounds", 3),
    )
    logger.info("Red Team Agent created (refine=%d rounds)", red_team.refinement_rounds)

    # ── SDE engine (semantic discovery + rescoring) ──
    from sde.engine import SemanticDiscoveryEngine
    sde_engine = SemanticDiscoveryEngine(
        convergence_std=0.05,
        max_rounds=25,
        use_composite=True,
    )
    victim_name = config.get("victim", {}).get("model_name", "model")
    sde_engine.initialise(victim_name)
    strategist.sde_engine = sde_engine
    strategist._semantic_enabled = True

    # ── Patch execute_intervention (sync) ──
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

    # ── Patch async_execute_intervention (async) ──
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
    logger.info("SDE engine integrated: semantic discovery + rescoring active (sync + async)")

    res_cfg = config["researcher"]
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

    # ── Orchestrator ──
    from orchestration.orchestrator import Orchestrator
    orchestrator = Orchestrator(
        cognitive_agent=cognitive,
        strategist_agent=strategist,
        researcher_agent=researcher,
        red_team_agent=red_team,
        knowledge_manager=km,
        session_memory=session,
        victim=victim,
        campaign_id=campaign_id,
        experiment_id=experiment_id,
        max_iterations=cfg["max_iterations"],
        max_interventions=cfg["max_interventions"],
        accuracy_threshold=cfg["accuracy_threshold"],
        error_tolerance=cfg.get("error_tolerance", 0.15),
        synthesis_interval=cfg["synthesis_interval"],
        force_exploration_interval=cfg.get("force_exploration_interval", 3),
        top_k_candidates=res_cfg.get("top_k_candidates", 30),
        seed_telemetry=seed_telemetry,
        holdout_prompts_path=cfg.get("holdout_prompts_path", None),
        convergence_config=cfg.get("convergence"),
        force_exploration_config=cfg.get("force_exploration"),
        min_interventions_per_iteration=cfg.get("min_interventions_per_iteration", 3),
        uncertainty_sampling_fallback=cfg.get("uncertainty_sampling_fallback", True),
        absolute_max_iterations=cfg.get("absolute_max_iterations", 200),
        exclude_prompts=test_prompts,
    )

    # ── Run ──
    logger.info("")
    logger.info("=" * 60)
    logger.info("Starting pipeline")
    logger.info("=" * 60)
    start = time.time()
    use_async = config.get("use_async", False)
    if use_async:
        logger.info("Using async pipeline (Orchestrator.async_run)")
        result = asyncio.run(orchestrator.async_run())
    else:
        result = orchestrator.run()
    elapsed = time.time() - start
    logger.info("Pipeline finished in %.1f s", elapsed)

    # ── Save SDE evidence ──
    try:
        import json as _json
        sde_ev = sde_engine.get_semantic_evidence()
        if sde_ev is not None:
            campaign_out = OUTPUTS_DIR / campaign_id
            campaign_out.mkdir(parents=True, exist_ok=True)
            sde_path = campaign_out / "sde_evidence.json"
            with open(sde_path, "w") as _f:
                _json.dump({
                    "sde_state": sde_engine.get_state().to_dict(),
                    "semantic_evidence": sde_ev,
                }, _f, indent=2, default=str)
            logger.info("SDE evidence saved to %s", sde_path)
    except Exception as e:
        logger.warning("Could not save SDE evidence: %s", e)

    # ── Save all learned campaign state ──
    from core.jailbreak import _TECHNIQUE_STATS
    from knowledge.campaign_state import save_campaign_state
    campaign_state_dir = save_campaign_state(
        output_dir=str(OUTPUTS_DIR),
        campaign_id=campaign_id,
        version_space=orchestrator.version_space,
        surrogate=orchestrator.surrogate,
        sde_engine=sde_engine,
        causal_graph=causal,
        technique_stats=_TECHNIQUE_STATS,
    )

    # ── Save outputs ──
    _save_outputs(result, defense, scientific, episodic, campaign_id, experiment_id)

    # ── Evaluation (RQ0, RQ1, ASR, Harmony ASR) ──
    try:
        _run_evaluation(
            result,
            victim,
            episodic,
            defense,
            campaign_id,
            experiment_id,
            harmful_csv,
            BENIGN_CSV,
            prior_campaign_id,
            llm=llm,
            red_team=red_team,
            knowledge_dir=campaign_state_dir,
            agentic_backend=agentic_backend,
            num_asr_str=num_asr_str,
            num_variants=num_variants,
        )
    except Exception as e:
        logger.warning("Evaluation failed (non-fatal): %s", e)

    # ── Cleanup ──
    for store in [episodic, defense, causal, ontology, scientific, session]:
        try:
            store.close()
        except Exception:
            pass

    return result


def _save_outputs(
    result: dict,
    defense: Any,
    scientific: Any,
    episodic: Any,
    campaign_id: str,
    experiment_id: str,
) -> None:
    campaign_out = OUTPUTS_DIR / campaign_id
    campaign_out.mkdir(parents=True, exist_ok=True)

    # 1. Final program (AST)
    best_id = result.get("best_program_id")
    if best_id:
        try:
            prog = defense.get(best_id)
            if prog is not None:
                prog_dict = prog.to_dict() if hasattr(prog, "to_dict") else prog
            else:
                vs = result.get("version_space", {})
                prog_dict = vs.get("program_asts", {}).get(best_id)
                if prog_dict:
                    logger.info("Fell back to version_space for program %s", best_id)
            if prog_dict:
                with open(campaign_out / "final_program.json", "w") as f:
                    json.dump(prog_dict, f, indent=2, default=str)
                logger.info("Saved program %s to final_program.json", best_id)
        except Exception as e:
            logger.warning("Could not save final program: %s", e)

    # 2. Final theory
    try:
        theory_export_path = campaign_out / "final_theory.json"
        scientific.export_theories(str(theory_export_path), include_history=True)
        logger.info("Saved theories to final_theory.json")
    except Exception as e:
        logger.warning("Could not save final theory: %s", e)

    # 3. Interventions history
    try:
        episodes = episodic.get_episodes_by_campaign(campaign_id)
        interventions = []
        for ep in episodes:
            interventions.append({
                "episode_id": ep.episode_id,
                "outcome": ep.outcome,
                "prompt": ep.intervention.prompt if ep.intervention else "",
                "final_prompt": ep.intervention.final_prompt if ep.intervention else "",
                "transforms": ep.intervention.transforms if ep.intervention else [],
                "created_at": str(ep.created_at),
            })
        with open(campaign_out / "interventions_history.json", "w") as f:
            json.dump(interventions, f, indent=2, default=str)
        logger.info("Saved %d interventions to interventions_history.json", len(interventions))
    except Exception as e:
        logger.warning("Could not save interventions: %s", e)

    # 4. Hypotheses history
    hypotheses_log = {
        "total_iterations": result.get("total_iterations", 0),
        "total_interventions": result.get("total_interventions", 0),
        "best_accuracy": result.get("best_accuracy", 0.0),
        "best_program_id": result.get("best_program_id"),
        "success": result.get("success", False),
        "error": result.get("error"),
    }
    with open(campaign_out / "hypotheses_history.json", "w") as f:
        json.dump(hypotheses_log, f, indent=2, default=str)
    logger.info("Saved experiment summary to hypotheses_history.json")


def _run_evaluation(
    result: dict,
    victim: Any,
    episodic: Any,
    defense: Any,
    campaign_id: str,
    experiment_id: str,
    csv_path: str = "",
    benign_csv: str = "",
    prior_campaign_id: Optional[str] = None,
    llm: Optional[Any] = None,
    red_team: Optional[Any] = None,
    knowledge_dir: Optional[str] = None,
    agentic_backend: str = "openai",
    num_asr_str: str = "40",
    num_variants: int = 5,
) -> None:
    from evaluation.judges import LLMJudge, RuleBasedJudge
    from evaluation.evaluators import (
        BaselineASREvaluator,
        HarmonyXASREvaluator,
        RQ0Evaluator,
        RQ1Evaluator,
        RQ2Evaluator,
    )
    from prompt_loader import load_prompts

    judge = LLMJudge(llm_client=llm, llm_backend=agentic_backend, fallback_judge=RuleBasedJudge())
    campaign_out = OUTPUTS_DIR / campaign_id
    eval_dir = campaign_out / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    report: Dict[str, Any] = {
        "campaign_id": campaign_id,
        "experiment_id": experiment_id,
    }

    # RQ0: Program accuracy on held-out test set (balanced harmful + benign)
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
                    logger.info("Reconstructed program %s from version_space for RQ0", best_id)
            if program is not None:
                harmful_test = load_prompts(csv_path=csv_path, n=25, seed=555)
                benign_test = load_prompts(csv_path=benign_csv, n=25, seed=555)
                test_prompts = harmful_test + benign_test
                rq0 = RQ0Evaluator(victim=victim, judge=judge)
                rq0_result = rq0.evaluate(program=program, test_prompts=test_prompts)
                report["rq0"] = rq0_result
            else:
                report["rq0"] = {"program_id": best_id, "accuracy": 0.0, "passed": False, "error": "Program not found in defense or version_space"}
        except Exception as e:
            logger.warning("RQ0 evaluation failed: %s", e)
            report["rq0"] = {"program_id": best_id, "error": str(e)}
    else:
        report["rq0"] = {"program_id": None, "accuracy": 0.0, "passed": False}

    # RQ1: Intervention efficiency — measure program prediction accuracy
    try:
        rq1_predict_fn = None
        best_id = result.get("best_program_id")
        if best_id:
            program = None
            record = defense.get(best_id)
            if record is not None:
                program = record.program if hasattr(record, "program") else record
            else:
                vs = result.get("version_space", {})
                prog_dict = vs.get("program_asts", {}).get(best_id)
                if prog_dict:
                    from core.program import Program as _Program
                    program = _Program.from_dict(prog_dict)
                    logger.info("Reconstructed program %s from version_space for RQ1", best_id)
            if program is not None:
                from core.executor import ProgramExecutor
                from core.primitive import PrimitiveRegistry
                _executor = ProgramExecutor(registry=PrimitiveRegistry())
                def rq1_predict_fn(prompt: str) -> int:
                    try:
                        return int(_executor.execute(program, prompt))
                    except Exception:
                        return 0
        from prompt_loader import load_prompts
        val_harmful = load_prompts(csv_path=csv_path, n=50, seed=777)
        val_benign = load_prompts(csv_path=benign_csv, n=50, seed=777)
        val_prompts = val_harmful + val_benign
        val_labels = [1] * len(val_harmful) + [0] * len(val_benign)
        test_harmful = load_prompts(csv_path=csv_path, n=50, seed=999)
        test_benign = load_prompts(csv_path=benign_csv, n=50, seed=999)
        test_prompts = test_harmful + test_benign
        test_labels = [1] * len(test_harmful) + [0] * len(test_benign)
        rq1 = RQ1Evaluator(episodic)
        rq1._metric.set_validation_set(val_prompts, val_labels)
        rq1._metric.set_test_set(test_prompts, test_labels)
        rq1_result = rq1.evaluate(
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            threshold=0.85,
            predict_fn=rq1_predict_fn,
        )
        report["rq1"] = rq1_result
    except Exception as e:
        logger.warning("RQ1 evaluation failed: %s", e)
        report["rq1"] = {"error": str(e)}

    # ASR: Attack Success Rate (baseline — raw prompts)
    # Resolve num_asr once for both baseline and harmony evaluations
    if num_asr_str == "full" and csv_path:
        from evaluation.utils.test_generator import TestGenerator
        tg = TestGenerator(csv_path)
        num_asr = len(tg._prompts)
    else:
        num_asr = int(num_asr_str)
    total_variants = num_asr * num_variants

    try:
        asr_eval = BaselineASREvaluator(victim=victim, judge=judge, csv_path=csv_path)
        asr_result = asr_eval.evaluate(num_prompts=total_variants)
        report["asr"] = asr_result
        logger.info("ASR (baseline): %.4f (%d/%d accepted)",
                    asr_result["asr"],
                    asr_result["successes"],
                    asr_result["total"])
    except Exception as e:
        logger.warning("ASR evaluation failed: %s", e)
        report["asr"] = {"error": str(e)}

    # HARMONY_X ASR: unified knowledge-aware attack pipeline
    try:
        harmonyx_asr_eval = HarmonyXASREvaluator(
            victim=victim, judge=judge, csv_path=csv_path,
            red_team_agent=red_team, num_variants=1,
            knowledge_dir=knowledge_dir,
        )
        harmonyx_asr_result = harmonyx_asr_eval.evaluate(
            num_prompts=total_variants, num_variants=num_variants,
        )
        report["harmonyx_asr"] = harmonyx_asr_result
        easr = harmonyx_asr_result.get("easr", harmonyx_asr_result["asr"])
        logger.info("HarmonyX ASR: asr=%.4f easr=%.4f (%d/%d) program_blocked=%d",
                    harmonyx_asr_result["asr"], easr,
                    harmonyx_asr_result["successes"],
                    harmonyx_asr_result["total"],
                    harmonyx_asr_result.get("program_blocked", 0))
    except Exception as e:
        logger.warning("HarmonyX ASR evaluation failed: %s", e)
        report["harmonyx_asr"] = {"error": str(e)}

    # RQ2: Transfer speed (requires a prior campaign for comparison)
    try:
        if prior_campaign_id:
            rq2 = RQ2Evaluator(
                episodic,
                db_dir=str(OUTPUTS_DIR),
                outputs_dir=str(OUTPUTS_DIR),
            )
            rq2_result = rq2.evaluate(
                prior_campaign_id=prior_campaign_id,
                target_campaign_id=campaign_id,
                prior_experiment_id=None,
                target_experiment_id=experiment_id,
                threshold=0.9,
            )
            report["rq2"] = rq2_result
            logger.info(
                "RQ2: prior=%s target=%s speedup=%.2f",
                prior_campaign_id, campaign_id,
                rq2_result.get("transfer_speedup", 0.0),
            )
        else:
            report["rq2"] = {"note": "No prior campaign provided for comparison"}
            logger.info("RQ2: Skipped (no prior_campaign_id)")
    except Exception as e:
        logger.warning("RQ2 evaluation failed (expected on first run): %s", e)
        report["rq2"] = {"error": str(e), "note": "Needs a prior campaign for meaningful comparison"}

    report_path = eval_dir / "evaluation_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Evaluation report saved to %s", report_path)


def print_report(result: dict, title: str = "HARMONY-X Experiment") -> None:
    logger.info("")
    logger.info("=" * 60)
    logger.info("  %s — Summary", title)
    logger.info("=" * 60)
    logger.info("  Result:          %s", "PASS" if result.get("success") else "FAIL")
    logger.info("  Best program ID: %s", result.get("best_program_id", "N/A"))
    logger.info("  Best accuracy:   %.4f", result.get("best_accuracy", 0.0))
    logger.info("  Interventions:   %d", result.get("total_interventions", 0))
    logger.info("  Iterations:      %d", result.get("total_iterations", 0))
    if result.get("error"):
        logger.info("  Error:           %s", result["error"])
    logger.info("=" * 60)

    # Anomaly-source telemetry
    telemetry = result.get("telemetry", {})
    anomaly_tel = telemetry.get("anomaly", {})
    if anomaly_tel.get("anomaly_count", 0) > 0:
        logger.info("  Anomaly telemetry:")
        logger.info("    Total anomalies:  %d", anomaly_tel["anomaly_count"])
        for family, count in sorted(anomaly_tel.get("anomaly_count_by_family", {}).items()):
            rate = anomaly_tel.get("anomaly_rate_by_family", {}).get(family, 0)
            logger.info("    %-25s %3d (%5.1f%%)", family, count, rate * 100)
        logger.info("    --- By source category ---")
        for src, count in sorted(anomaly_tel.get("anomaly_count_by_source", {}).items()):
            logger.info("    %-30s %3d", src, count)
        logger.info("=" * 60)

    cid = result.get("campaign_id")
    prog_path = (OUTPUTS_DIR / cid / "final_program.json") if cid else (OUTPUTS_DIR / "final_program.json")
    if prog_path.exists():
        with open(prog_path) as f:
            prog_data = json.load(f)
        logger.info("")
        logger.info("Final program (AST):")
        logger.info(json.dumps(prog_data, indent=2))


# ── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HARMONY-X experiment: reverse-engineer LLM safety layer",
    )
    parser.add_argument(
        "--config", default=None,
        help="Path to config YAML (default: configs/<backend>_experiment_config.yaml)",
    )
    parser.add_argument(
        "--num-seeds", type=int, default=100,
        help="Number of seed prompts to load (default: 100, use --full for all)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use all prompts from CSVs (overrides --num-seeds)",
    )
    parser.add_argument(
        "--prior-campaign", type=str, default=None,
        help="Prior campaign ID for RQ2 transfer speed evaluation",
    )
    parser.add_argument(
        "--backend", choices=["openrouter", "ollama", "openai"], default="ollama",
        help="Backend to use for the victim LLM (default: ollama)",
    )
    parser.add_argument(
        "--agentic-backend", choices=["openrouter", "openai"], default="openai",
        help="Backend for cognitive/red-team/judge LLMs (default: openai)",
    )
    parser.add_argument(
        "--campaign-prefix", type=str, default=None,
        help="Campaign prefix (default: derived from config model_name)",
    )
    parser.add_argument(
        "--model-name", type=str, default=None,
        help="Override victim.model_name in config",
    )
    parser.add_argument(
        "--num-asr", type=str, default="40",
        help="Number of original prompts for ASR evaluation "
             "(default: 40, use 'full' for all prompts in CSV)",
    )
    parser.add_argument(
        "--num-variants", type=int, default=5,
        help="Number of variants per original prompt in ASR evaluation (default: 5)",
    )
    args = parser.parse_args()

    config_path = args.config or str(EXP_DIR / "configs" / f"{args.backend}_experiment_config.yaml")
    config = load_config(config_path)
    config["num_seeds"] = args.num_seeds
    config["full_seeds"] = args.full

    if args.model_name:
        config["victim"]["model_name"] = args.model_name

    # Setup logging
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = str(LOGS_DIR / f"experiment_{timestamp}.log")
    setup_logging(log_file)

    model_name = config.get("victim", {}).get("model_name", "model")
    campaign_prefix = args.campaign_prefix or re.sub(r'[^a-zA-Z0-9_]+', '_', model_name).strip('_')

    logger.info("=" * 60)
    logger.info("HARMONY-X — %s experiment", campaign_prefix)
    logger.info("=" * 60)
    agentic_backend = args.agentic_backend
    logger.info("Config: %s", config_path)
    logger.info("Backend: %s", args.backend)
    logger.info("Agentic backend: %s", agentic_backend)
    logger.info("Log:    %s", os.path.abspath(log_file))
    logger.info("")

    result = run_experiment(config, backend=args.backend,
                            campaign_prefix=args.campaign_prefix,
                            prior_campaign_id=args.prior_campaign,
                            agentic_backend=agentic_backend,
                            num_asr_str=args.num_asr,
                            num_variants=args.num_variants,)

    title = args.campaign_prefix or f"HARMONY-X — {model_name} experiment"
    print_report(result, title=title)

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
