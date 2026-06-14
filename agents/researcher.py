"""Researcher Agent — reverse engineering safety programs through intervention-guided synthesis.

The Researcher Agent is the core of HARMONY-X's reverse engineering pipeline.
It reads intervention data from Episodic Memory, synthesizes programs, verifies
them against a victim, stores verified programs in the Defense Program Store,
extracts abstract theories, and persists them in Scientific Memory.

This agent does NOT use LLMs — it relies entirely on the synthesis module
and the hierarchical memory layers.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from core.executor import ProgramExecutor
from core.primitive import default_registry
from core.program import Program
from core.types import Outcome
from knowledge.defense_store import DefenseProgramRecord, DefenseProgramStore
from knowledge.episodic.episodic import EpisodicMemory, EpisodeFilter
from knowledge.ontology_memory import OntologyMemory
from graphrag.graph_reasoner import GraphRAGAnswer, GraphReasoner
from graphrag.query_parser import QueryParser
from graphrag.subgraph_retriever import SubgraphRetriever
from knowledge.scientific_memory import ScientificMemory, Theory
from knowledge.defense_store import DefenseProgramRecord, DefenseProgramStore
from harmony.synthesis import get_synthesizer, SynthesisStats
from synthesis.preprocessor import Preprocessor
from synthesis.verifier import ProgramVerifier, VerificationReport

logger = logging.getLogger(__name__)


class ResearcherAgent:
    """Orchestrates program synthesis from episodic data, verification, and knowledge persistence.

    The Researcher Agent executes the core reverse engineering pipeline:
        1. Read intervention data from Episodic Memory
        2. Synthesize a program from examples using the synthesizer
        3. Verify the program against a victim via ProgramVerifier
        4. Store the verified program in Defense Program Store
        5. Extract an abstract theory from the program and save to Scientific Memory

    Attributes:
        episodic_memory: Episodic Memory (L1) for reading intervention records.
        defense_store: Defense Program Store (L4) for persisting verified programs.
        scientific_memory: Scientific Memory (L6) for persisting abstract theories.
        ontology_memory: Optional Ontology Memory (L5) for primitive catalog.
        synthesizer: Program synthesizer used to generate programs from examples.
        executor: Program executor used to evaluate programs.
        verifier: Optional shared ProgramVerifier. Created per verify_program() call if not set.
        default_model_family: Default model family label for abstract theories.
        causal_graph: Optional Causal Graph (K2) for guiding synthesis.
        _verifier_cache: Cache of {victim_id: ProgramVerifier} for verifier reuse.
    """

    def __init__(
        self,
        episodic_memory: EpisodicMemory,
        defense_store: DefenseProgramStore,
        scientific_memory: ScientificMemory,
        ontology_memory: Optional[OntologyMemory] = None,
        synthesizer: Optional[Any] = None,
        executor: Optional[ProgramExecutor] = None,
        verifier: Optional[ProgramVerifier] = None,
        default_model_family: str = "unknown",
        causal_graph: Optional[Any] = None,
    ) -> None:
        """Initialize the Researcher Agent.

        Args:
            episodic_memory: Episodic Memory instance for reading intervention data.
            defense_store: Defense Program Store for persisting programs.
            scientific_memory: Scientific Memory for persisting theories.
            ontology_memory: Optional Ontology Memory for primitive catalog.
            synthesizer: Optional synthesizer instance. Created with defaults if not provided.
            executor: Optional ProgramExecutor. Created from default_registry if not provided.
            verifier: Optional ProgramVerifier. Created per verify_program() call if not set.
            default_model_family: Default model family label used when abstracting theories.
            causal_graph: Optional Causal Graph (K2) instance for guiding synthesis.

        Raises:
            TypeError: If episodic_memory, defense_store, or scientific_memory are None.
        """
        self.episodic_memory = episodic_memory
        self.defense_store = defense_store
        self.scientific_memory = scientific_memory
        self.ontology_memory = ontology_memory
        self.causal_graph = causal_graph

        # GraphRAG for explanatory reasoning (Section 10)
        self._query_parser = QueryParser()
        self._subgraph_retriever: Optional[SubgraphRetriever] = None
        self._graph_reasoner = GraphReasoner()
        self._init_graphrag()



        self.synthesizer = synthesizer or get_synthesizer(
            mode="evolutionary",
            config={"population_size": 100, "generations": 30, "mutation_rate": 0.2, "crossover_rate": 0.7},
        )
        self.executor = executor or ProgramExecutor(default_registry)
        self._shared_verifier = verifier
        self.default_model_family = default_model_family
        self._verifier_cache: Dict[str, ProgramVerifier] = {}
        self.preprocessor = Preprocessor()

    # ------------------------------------------------------------------
    # Step 1: Synthesis
    # ------------------------------------------------------------------

    def synthesize_from_campaign(
        self,
        campaign_id: str,
        experiment_id: Optional[str] = None,
        error_tolerance: float = 0.15,
        exclude_prompts: Optional[Set[str]] = None,
        exclude_episode_ids: Optional[Set[str]] = None,
    ) -> Tuple[Optional[Program], SynthesisStats]:
        """Synthesize a program from episodes of a given campaign.

        Reads all episodes matching the campaign (and optionally experiment) from
        Episodic Memory, converts each to a (prompt, outcome) example, and runs
        the synthesizer.

        Args:
            campaign_id: Campaign ID to read episodes from.
            experiment_id: Optional experiment ID to narrow the scope.
            error_tolerance: Noise tolerance (0.0..1.0). Clamped to [0.0, 1.0] if out of range.
            exclude_prompts: Optional set of prompt strings to exclude from training examples.
            exclude_episode_ids: Optional set of episode IDs to exclude from training examples.

        Returns:
            Tuple of (program or None if synthesis failed, SynthesisStats).

        Raises:
            ValueError: If error_tolerance is outside [0.0, 1.0] (logged as warning, clamped).
        """
        error_tolerance = self._validate_error_tolerance(error_tolerance)

        filter_kwargs: Dict[str, Any] = {"campaign_id": campaign_id}
        if experiment_id is not None:
            filter_kwargs["experiment_id"] = experiment_id

        episode_filter = EpisodeFilter(**filter_kwargs)
        episodes = self.episodic_memory.filter_episodes(episode_filter)

        if not episodes:
            logger.warning(
                "No episodes found for campaign=%s experiment=%s",
                campaign_id, experiment_id,
            )
            return None, SynthesisStats()

        examples: List[Tuple[str, int]] = []
        excluded_by_prompt = 0
        excluded_by_id = 0
        for ep in episodes:
            if exclude_episode_ids and ep.episode_id in exclude_episode_ids:
                excluded_by_id += 1
                continue
            prompt = ep.intervention.final_prompt or ep.intervention.prompt
            if exclude_prompts and prompt in exclude_prompts:
                excluded_by_prompt += 1
                continue
            if prompt and ep.outcome is not None:
                examples.append((prompt, int(ep.outcome)))

        # Run preprocessor (normalize + denoise)
        examples = self.preprocessor.process(examples)

        log_extra = ""
        if excluded_by_prompt or excluded_by_id:
            log_extra = f" (excluded: {excluded_by_prompt} prompts, {excluded_by_id} ids)"
        if not examples:
            logger.warning(
                "No valid examples after exclude filters for campaign=%s%s",
                campaign_id, log_extra,
            )
            return None, SynthesisStats()

        logger.info(
            "Synthesizing from %d examples (campaign=%s, experiment=%s, error_tolerance=%.2f%s)",
            len(examples), campaign_id, experiment_id, error_tolerance, log_extra,
        )

        start_ts = time.time()
        candidates = self.synthesizer.synthesize(examples, k=1)
        elapsed = time.time() - start_ts

        program = candidates[0] if candidates else None
        stats = SynthesisStats()
        stats.duration_ms = elapsed * 1000
        stats.programs_tried = len(candidates)
        stats.synthesized_candidates = len(candidates)
        stats.candidates_considered = len(candidates) * self.synthesizer.generations if hasattr(self.synthesizer, 'generations') else len(candidates)

        if program is not None:
            logger.info(
                "Synthesis succeeded in %.1fs (candidates=%d)",
                elapsed, stats.synthesized_candidates,
            )
        else:
            logger.warning(
                "Synthesis failed for campaign=%s after %.1fs",
                campaign_id, elapsed,
            )

        return program, stats

    def synthesize_program(
        self,
        examples: List[Tuple[str, int]],
        grammar: Optional[Any] = None,
    ) -> Tuple[Optional[Program], SynthesisStats]:
        """Standalone synthesis from raw examples (spec §5.3).

        Parameters
        ----------
        examples : list of (prompt, outcome) tuples.
        grammar : optional grammar object (currently unused; the synthesizer
                  uses the default primitive registry and ontology).

        Returns
        -------
        Tuple of (program or None, SynthesisStats).
        """
        examples = self.preprocessor.process(examples)
        if not examples:
            logger.warning("All examples removed by preprocessor")
            return None, SynthesisStats()
        candidates = self.synthesizer.synthesize(examples, k=1)
        program = candidates[0] if candidates else None
        stats = SynthesisStats()
        stats.synthesized_candidates = len(candidates)
        stats.programs_tried = len(candidates)
        return program, stats

    # ------------------------------------------------------------------
    # Step 2: Verification
    # ------------------------------------------------------------------

    def verify_program(
        self,
        program: Program,
        victim: Any,
        num_test_interventions: int = 10,
        accuracy_threshold: float = 0.9,
        exclude_prompts: Optional[Set[str]] = None,
        verbose: bool = False,
    ) -> VerificationReport:
        """Verify a synthesized program against a victim via ProgramVerifier.

        Uses a cached ProgramVerifier per victim if available, otherwise creates a new one.

        Args:
            program: The program to verify.
            victim: A victim instance compatible with ProgramVerifier.
            num_test_interventions: Number of test interventions to generate.
            accuracy_threshold: Minimum accuracy required for verification.
            exclude_prompts: Optional set of prompts to exclude from testing.
            verbose: If True, enable verbose logging during verification.

        Returns:
            VerificationReport with accuracy, failures, suggestions.

        Raises:
            TypeError: If program or victim are None.
        """
        victim_id = getattr(victim, "victim_id", None) or str(id(victim))
        verifier = self._get_verifier(victim)

        start_ts = time.time()
        report = verifier.verify(
            program,
            num_test_interventions=num_test_interventions,
            accuracy_threshold=accuracy_threshold,
            exclude_prompts=exclude_prompts,
            verbose=verbose,
        )
        elapsed = time.time() - start_ts

        logger.info(
            "Verification completed in %.1fs (victim=%s, accuracy=%.2f, verified=%s, failures=%d, tested=%d)",
            elapsed, victim_id, report.accuracy, report.verified,
            len(report.failures), report.num_tested,
        )
        return report

    # ------------------------------------------------------------------
    # Step 3: Store program
    # ------------------------------------------------------------------

    def store_program(
        self,
        program: Program,
        name: str = "",
        confidence: float = 0.0,
        provenance: Optional[List[str]] = None,
        status: str = "confirmed",
    ) -> str:
        """Store a verified program in the Defense Program Store.

        Args:
            program: The program to store.
            name: Human-readable name for the program.
            confidence: Confidence score (0.0..1.0).
            provenance: List of evidence/experiment IDs supporting this program.
            status: Status label (e.g. "confirmed", "draft", "rejected").

        Returns:
            The stored program's ID.
        """
        metadata: Dict[str, Any] = {"status": status}
        record = DefenseProgramRecord(
            id=name or program.id or "",
            name=name or f"synth_{program.id}",
            program=program,
            confidence=confidence,
            provenance=provenance or [],
            metadata=metadata,
        )
        program_id = self.defense_store.save(record)
        logger.info(
            "Stored program id=%s name='%s' confidence=%.2f status='%s'",
            program_id, record.name, record.confidence, status,
        )
        return program_id

    # ------------------------------------------------------------------
    # Step 4: Abstract theory
    # ------------------------------------------------------------------

    def abstract_theory(
        self,
        program: Program,
        model_family: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        provenance: Optional[List[str]] = None,
    ) -> Theory:
        """Extract an abstract theory from a program.

        Uses the synthesizer's abstract_theory method if available, otherwise
        constructs a Theory from the program's string representation.

        Args:
            program: The program to abstract.
            model_family: Model family label (defaults to self.default_model_family).
            conditions: Additional conditions for the theory.
            provenance: List of evidence/experiment IDs.

        Returns:
            A Theory dataclass instance.

        Raises:
            TypeError: If program is None.
        """
        theory = self._do_abstract_theory(
            program,
            model_family=model_family or self.default_model_family,
            conditions=conditions,
            provenance=provenance,
        )
        pattern_preview = theory.pattern[:80] if len(theory.pattern) > 80 else theory.pattern
        logger.info(
            "Abstracted theory id=%s pattern='%s'", theory.id, pattern_preview,
        )
        return theory

    # ------------------------------------------------------------------
    # Step 5: Store theory
    # ------------------------------------------------------------------

    def store_theory(self, theory: Theory) -> str:
        """Store an abstract theory in Scientific Memory.

        Args:
            theory: The Theory instance to store.

        Returns:
            The stored theory's ID.
        """
        theory_id = self.scientific_memory.save_theory(theory)
        logger.info("Stored theory id=%s", theory_id)
        return theory_id

    def store_scientific_knowledge(self, theory: Theory) -> str:
        """Alias for store_theory — spec-compatible name (Section 5.3)."""
        return self.store_theory(theory)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def verify_and_store(
        self,
        program: Program,
        campaign_id: str,
        victim: Any,
        program_id: Optional[str] = None,
        accuracy_threshold: float = 0.8,
    ) -> Tuple[bool, float]:
        """Verify a candidate program against victim and store if verified.

        Returns (True, accuracy) if program was verified and stored successfully,
        or (False, 0.0) otherwise.
        """
        try:
            report = self.verify_program(
                program=program,
                victim=victim,
                num_test_interventions=50,
                accuracy_threshold=accuracy_threshold,
            )
            if report.verified:
                stored_id = self.store_program(
                    program=program,
                    name=program_id or program.id,
                    confidence=report.accuracy,
                    provenance=[campaign_id],
                    status="verified" if report.accuracy >= accuracy_threshold else "candidate",
                )
                logger.info(
                    "Verified and stored program %s (accuracy=%.2f)",
                    stored_id, report.accuracy,
                )
                return True, report.accuracy
            return False, 0.0
        except Exception as exc:
            logger.debug("verify_and_store failed: %s", exc)
            return False, 0.0

    def run_reverse_engineering_pipeline(
        self,
        campaign_id: str,
        victim: Any,
        program_name: Optional[str] = None,
        error_tolerance: float = 0.15,
        num_test_interventions: int = 10,
        accuracy_threshold: float = 0.9,
        model_family: Optional[str] = None,
        experiment_id: Optional[str] = None,
        verbose: bool = False,
        exclude_prompts: Optional[Set[str]] = None,
        exclude_episode_ids: Optional[Set[str]] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Run the full reverse engineering pipeline end-to-end.

        Executes:
            1. synthesize_from_campaign — read episodes and synthesize program
            2. verify_program — verify against victim
            3. store_program — save verified program to Defense Program Store
            4. abstract_theory — extract theory from the program
            5. store_theory — persist theory to Scientific Memory

        Supports checkpoint resume: if a checkpoint dict is provided with a
        completed step, that step is skipped.

        Args:
            campaign_id: Campaign ID to process.
            victim: Victim instance for verification.
            program_name: Optional name for the stored program.
            error_tolerance: Noise tolerance for synthesis. Clamped to [0.0, 1.0].
            num_test_interventions: Number of test interventions for verification.
            accuracy_threshold: Accuracy threshold for verification.
            model_family: Model family for theory abstraction.
            experiment_id: Optional experiment ID to narrow the scope.
            verbose: If True, enable verbose verification logging.
            exclude_prompts: Optional set of prompt strings to exclude from training examples.
            exclude_episode_ids: Optional set of episode IDs to exclude from training examples.
            checkpoint: Optional dict with keys {"program", "program_id", "theory", "step"}
                to resume from a previous partial run.

        Returns:
            Dict with keys:
                - success (bool): Whether the pipeline completed.
                - program_id (str or None): ID of stored program.
                - theory_id (str or None): ID of stored theory.
                - accuracy (float): Verification accuracy.
                - verified (bool): Whether verification passed threshold.
                - stats (SynthesisStats or None): Synthesis statistics.
                - report (VerificationReport or None): Verification report.
                - error (str or None): Error message if any.
                - checkpoint (dict): Intermediate state for resume support.

        Raises:
            TypeError: If campaign_id or victim are None.
        """
        error_tolerance = self._validate_error_tolerance(error_tolerance)
        start_time = time.time()
        result: Dict[str, Any] = self._init_pipeline_result()

        checkpoint = checkpoint or {}
        step_start = checkpoint.get("step", 0)

        try:
            # Step 1: Synthesis
            if step_start <= 1:
                program, stats = self._pipeline_step1_synthesis(
                    campaign_id, experiment_id, error_tolerance,
                    exclude_prompts, exclude_episode_ids,
                )
                result["stats"] = stats
                if program is None:
                    return self._finalize_pipeline_result(
                        result, start_time, pipeline_step=1,
                    )
            else:
                program = checkpoint.get("program")
                logger.info("Checkpoint resume: skipping synthesis (step 1)")

            # Step 2: Verification
            if step_start <= 2:
                report = self._pipeline_step2_verification(
                    program, victim, num_test_interventions,
                    accuracy_threshold, verbose,
                )
                result["report"] = report
                result["accuracy"] = report.accuracy
                result["verified"] = report.verified
            else:
                report = result.get("report")
                logger.info("Checkpoint resume: skipping verification (step 2)")

            # Step 3: Store program
            if step_start <= 3:
                provenance = self._build_provenance(campaign_id, experiment_id)
                program_id = self._pipeline_step3_store(
                    program, program_name, report, provenance,
                )
                result["program_id"] = program_id
            else:
                program_id = checkpoint.get("program_id") or result.get("program_id")
                logger.info("Checkpoint resume: skipping store_program (step 3)")

            # Step 4: Abstract theory
            if step_start <= 4:
                conditions = self._build_theory_conditions(
                    campaign_id, experiment_id, report,
                )
                theory = self._pipeline_step4_theory(
                    program, model_family, conditions, provenance,
                )
            else:
                theory = checkpoint.get("theory")
                logger.info("Checkpoint resume: skipping abstract_theory (step 4)")

            # Step 5: Store theory
            if step_start <= 5:
                theory_id = self._pipeline_step5_store_theory(theory)
                result["theory_id"] = theory_id
            else:
                theory_id = result.get("theory_id")
                logger.info("Checkpoint resume: all steps already completed")

            result["success"] = True
            elapsed = time.time() - start_time
            logger.info(
                "Pipeline completed in %.1fs: campaign=%s program=%s theory=%s accuracy=%.2f verified=%s",
                elapsed, campaign_id, program_id or "(none)",
                theory_id or "(none)", result.get("accuracy", 0.0), result.get("verified", False),
            )

        except Exception as exc:
            logger.exception("Pipeline failed with exception")
            result["error"] = str(exc)

        return self._finalize_pipeline_result(result, start_time, pipeline_step=5)

    # ------------------------------------------------------------------
    # Knowledge Manager integration
    # ------------------------------------------------------------------

    def process_proposals(
        self,
        proposals: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Process proposals from other agents via the Knowledge Manager.

        Each proposal is a dict with keys:
            - type: str — proposal type (e.g. "synthesize", "verify", "store")
            - campaign_id: str
            - experiment_id: str (optional)
            - payload: dict (optional)

        Args:
            proposals: List of proposal dicts to process.

        Returns:
            List of result dicts, one per proposal, with keys:
                - success (bool): Whether processing succeeded.
                - proposal_type (str): The type of proposal.
                - result (Any): The result of processing.
                - error (str or None): Error message if any.
        """
        results: List[Dict[str, Any]] = []
        for proposal in proposals:
            ptype = proposal.get("type", "")
            result_entry: Dict[str, Any] = {
                "proposal_type": ptype,
                "success": False,
                "result": None,
                "error": None,
            }
            try:
                if ptype == "synthesize":
                    prog, stats = self.synthesize_from_campaign(
                        campaign_id=proposal["campaign_id"],
                        experiment_id=proposal.get("experiment_id"),
                        error_tolerance=proposal.get("error_tolerance", 0.15),
                    )
                    result_entry["result"] = (prog, stats)
                    result_entry["success"] = prog is not None
                elif ptype == "verify":
                    report = self.verify_program(
                        program=proposal["program"],
                        victim=proposal["victim"],
                    )
                    result_entry["result"] = report
                    result_entry["success"] = True
                elif ptype == "store_program":
                    pid = self.store_program(
                        program=proposal["program"],
                        name=proposal.get("name", ""),
                        confidence=proposal.get("confidence", 0.0),
                        provenance=proposal.get("provenance"),
                        status=proposal.get("status", "confirmed"),
                    )
                    result_entry["result"] = pid
                    result_entry["success"] = True
                elif ptype == "store_theory":
                    tid = self.store_theory(proposal["theory"])
                    result_entry["result"] = tid
                    result_entry["success"] = True
                else:
                    result_entry["error"] = f"Unknown proposal type: {ptype}"
            except Exception as exc:
                logger.exception("Proposal failed: type=%s", ptype)
                result_entry["error"] = str(exc)
            results.append(result_entry)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_abstract_theory(
        self,
        program: Program,
        model_family: str,
        conditions: Optional[Dict[str, Any]] = None,
        provenance: Optional[List[str]] = None,
    ) -> Theory:
        """Internal: attempt synthesizer.abstract_theory with fallback.

        Enhanced: queries Defense Program Store for sibling programs in the
        same model family, compares structures, and generalises the theory
        (spec §5.3: 'So sánh với các chương trình đã có cùng họ mô hình,
        tổng quát hóa thành Theory').
        """
        # Get base theory from synthesizer
        try:
            base_theory = self.synthesizer.abstract_theory(
                program,
                model_family=model_family,
                conditions=conditions,
                provenance=provenance,
            )
        except (AttributeError, Exception) as exc:
            logger.warning(
                "synthesizer.abstract_theory raised %s; building fallback Theory",
                exc,
            )
            pattern = str(program)
            fallback_conditions = dict(conditions or {})
            fallback_conditions.setdefault("model_family", model_family)
            base_theory = Theory(
                pattern=pattern,
                conditions=fallback_conditions,
                provenance=provenance or [f"fallback:{pattern[:64]}"],
            )

        # Cross-program comparison: find sibling programs in same model family
        sibling_patterns: List[str] = []
        try:
            pids = self.defense_store.list_program_ids()
            base_str = str(program)
            for pid in pids:
                try:
                    rec = self.defense_store.get(pid)
                    if rec is None:
                        continue
                    p = getattr(rec, "program", None)
                    if p is not None and str(p) != base_str:
                        p_meta = getattr(rec, "metadata", {}) or {}
                        if p_meta.get("model_family") == model_family:
                            sibling_patterns.append(str(p))
                except Exception:
                    continue
        except Exception as exc:
            logger.debug("Could not query sibling programs: %s", exc)

        if sibling_patterns:
            # Generalize: mark that this pattern was observed across N programs
            generalised_pattern = base_theory.pattern
            confidence_boost = min(0.2, len(sibling_patterns) * 0.05)
            base_theory.confidence = min(1.0, base_theory.confidence + confidence_boost)
            base_theory.conditions["num_sibling_programs"] = len(sibling_patterns)
            base_theory.conditions["sibling_patterns_sample"] = sibling_patterns[:3]
            base_theory.pattern = (
                f"[Generalised from {1 + len(sibling_patterns)} programs] "
                f"{generalised_pattern}"
            )
            logger.info(
                "Theory generalised: %d sibling programs found in model_family='%s' "
                "(confidence +%.2f)",
                len(sibling_patterns), model_family, confidence_boost,
            )

        return base_theory

    def _get_verifier(self, victim: Any) -> ProgramVerifier:
        """Return a cached or new ProgramVerifier for the given victim."""
        if self._shared_verifier is not None:
            return self._shared_verifier

        victim_id = getattr(victim, "victim_id", None) or str(id(victim))
        if victim_id not in self._verifier_cache:
            self._verifier_cache[victim_id] = ProgramVerifier(
                executor=self.executor, victim=victim,
            )
            logger.debug("Created cached verifier for victim=%s", victim_id)
        return self._verifier_cache[victim_id]

    @staticmethod
    def _validate_error_tolerance(rate: float) -> float:
        """Validate and clamp error_tolerance to [0.0, 1.0]."""
        if rate < 0.0 or rate > 1.0:
            logger.warning(
                "error_tolerance=%.2f outside [0, 1]; clamping to [0, 1]", rate,
            )
        return max(0.0, min(1.0, rate))

    @staticmethod
    def _init_pipeline_result() -> Dict[str, Any]:
        return {
            "success": False,
            "program_id": None,
            "theory_id": None,
            "accuracy": 0.0,
            "verified": False,
            "stats": None,
            "report": None,
            "error": None,
            "checkpoint": {"step": 0},
        }

    def _pipeline_step1_synthesis(
        self,
        campaign_id: str,
        experiment_id: Optional[str],
        error_tolerance: float,
        exclude_prompts: Optional[Set[str]],
        exclude_episode_ids: Optional[Set[str]],
    ) -> Tuple[Optional[Program], SynthesisStats]:
        logger.info("Pipeline step 1/5: synthesizing from campaign=%s", campaign_id)
        return self.synthesize_from_campaign(
            campaign_id=campaign_id,
            experiment_id=experiment_id,
            error_tolerance=error_tolerance,
            exclude_prompts=exclude_prompts,
            exclude_episode_ids=exclude_episode_ids,
        )

    def _pipeline_step2_verification(
        self,
        program: Program,
        victim: Any,
        num_test_interventions: int,
        accuracy_threshold: float,
        verbose: bool,
    ) -> VerificationReport:
        logger.info("Pipeline step 2/5: verifying program")
        return self.verify_program(
            program=program,
            victim=victim,
            num_test_interventions=num_test_interventions,
            accuracy_threshold=accuracy_threshold,
            verbose=verbose,
        )

    def _pipeline_step3_store(
        self,
        program: Program,
        program_name: Optional[str],
        report: VerificationReport,
        provenance: List[str],
    ) -> str:
        logger.info("Pipeline step 3/5: storing program")
        return self.store_program(
            program=program,
            name=program_name or f"synth_{program.id}",
            confidence=report.accuracy,
            provenance=provenance,
            status="confirmed" if report.verified else "draft",
        )

    def _pipeline_step4_theory(
        self,
        program: Program,
        model_family: Optional[str],
        conditions: Dict[str, Any],
        provenance: List[str],
    ) -> Theory:
        logger.info("Pipeline step 4/5: abstracting theory")
        return self.abstract_theory(
            program=program,
            model_family=model_family,
            conditions=conditions,
            provenance=provenance,
        )

    def _pipeline_step5_store_theory(self, theory: Theory) -> str:
        logger.info("Pipeline step 5/5: storing theory")
        return self.store_theory(theory)

    def _build_provenance(
        self,
        campaign_id: str,
        experiment_id: Optional[str],
    ) -> List[str]:
        provenance: List[str] = []
        if experiment_id:
            provenance.append(f"exp:{experiment_id}")
        try:
            episodes = self.episodic_memory.filter_episodes(
                EpisodeFilter(campaign_id=campaign_id)
            )
            provenance.extend(
                f"ep:{ep.episode_id}" for ep in episodes[:5]
            )
        except Exception:
            pass
        return provenance

    @staticmethod
    def _build_theory_conditions(
        campaign_id: str,
        experiment_id: Optional[str],
        report: VerificationReport,
    ) -> Dict[str, Any]:
        return {
            "campaign_id": campaign_id,
            "experiment_id": experiment_id or "",
            "accuracy": report.accuracy,
            "verified": report.verified,
        }

    def _finalize_pipeline_result(
        self,
        result: Dict[str, Any],
        start_time: float,
        pipeline_step: int = 0,
    ) -> Dict[str, Any]:
        """Update checkpoint info and handle early termination."""
        if pipeline_step < 5 and result["error"] is None:
            result["checkpoint"] = {"step": pipeline_step}
        if result["error"]:
            msg = result["error"]
        elif pipeline_step < 5:
            msg = f"Pipeline terminated at step {pipeline_step}"
            result["error"] = msg
        result["elapsed"] = time.time() - start_time
        return result

    def get_verifier_cache_info(self) -> Dict[str, int]:
        """Return info about the verifier cache for monitoring."""
        return {
            "cached_victims": len(self._verifier_cache),
            "has_shared_verifier": self._shared_verifier is not None,
        }

    def clear_verifier_cache(self) -> None:
        """Clear the verifier cache."""
        self._verifier_cache.clear()
        logger.info("Verifier cache cleared")

    # ------------------------------------------------------------------
    # GraphRAG integration (Section 10)
    # ------------------------------------------------------------------

    def _init_graphrag(self) -> None:
        """Lazy-init the SubgraphRetriever if causal_graph and defense_store are available."""
        if self.causal_graph is not None and self.defense_store is not None:
            self._subgraph_retriever = SubgraphRetriever(
                causal_graph=self.causal_graph,
                defense_store=self.defense_store,
            )
            logger.info("GraphRAG SubgraphRetriever initialised")

    def explain(self, question: str) -> GraphRAGAnswer:
        """Answer a "why" question using GraphRAG reasoning.

        Pipeline:
          1. Parse the natural-language question into a structured query.
          2. Retrieve the relevant subgraph from Causal Graph + Defense Program Store.
          3. Reason over the subgraph to produce an explanatory answer.

        Parameters
        ----------
        question : str
            A natural-language question (e.g. "Why does ROT13 bypass the filter?").

        Returns
        -------
        GraphRAGAnswer
            The answer with evidence nodes/edges and a confidence score.
        """
        query = self._query_parser.parse(question)

        if self._subgraph_retriever is not None:
            subgraph = self._subgraph_retriever.retrieve(query)
        else:
            logger.warning("GraphRAG not fully initialised; returning empty subgraph")
            subgraph = {"nodes": [], "edges": [], "query": question}

        answer = self._graph_reasoner.reason(subgraph)
        logger.info(
            "GraphRAG explanation: query='%s' answer='%s' confidence=%.2f",
            question[:60], answer.answer[:80], answer.confidence,
        )
        return answer
