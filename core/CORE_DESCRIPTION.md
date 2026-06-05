# HARMONY-X Core Module Description

## Overview
This document describes the `core` package implemented for HARMONY-X. It includes foundational runtime structures for representing and executing a Defense Program, along with helper utilities and data model classes for interventions, observations, and hypotheses.

## Package Structure
- `core/__init__.py`
- `core/types.py`
- `core/primitive.py`
- `core/program.py`
- `core/executor.py`
- `core/intervention.py`
- `core/observation.py`
- `core/hypothesis.py`
- `core/grammar.py`
- `core/utils.py`

## Implemented Components

### `core/types.py`
Defines shared type aliases used across the core package:
- `Prompt = str`
- `Outcome = Literal[0, 1]`
- `Confidence = float`
- `ProgramID`, `InterventionID`, `ObservationID`, `HypothesisID` as `str`
- `Timestamp = float`

### `core/primitive.py`
Implements a primitive system for building programs from generic building blocks.
- Abstract base class `Primitive` with `name`, `parameters`, and `evaluate()`.
- Specialized primitive classes: `Predicate`, `Transform`, `Classifier`.
- Example primitives:
  - `ContainsWordPredicate`
  - `LengthGtPredicate`
  - `MatchesRegexPredicate`
  - `Rot13Transform`
  - `Base64DecodeTransform`
  - `ToLowercaseTransform`
  - `RemovePunctuationTransform`
  - `ToxicityScoreClassifier`
- `PrimitiveRegistry` singleton for register/get/list support.
- Default registry instance is initialized with sample primitives.

### `core/program.py`
Defines the Defense Program AST and serialization logic.
- Abstract `Node` base class.
- Atomic nodes: `PredicateNode`, `TransformNode`, `ClassifierNode`.
- Composite nodes: `AndNode`, `OrNode`, `NotNode`, `IfThenElseNode`, `ThresholdNode`, `ApplyTransformNode`.
- `Program` wrapper around an `IfThenElseNode` root, including versioning, metadata, canonicalization, and complexity metrics.
- `ProgramFragment` for reusable subgraphs.
- `Policy` and `PolicyTemplate` abstractions for independent policy objects and instantiable templates.
- Support for `to_dict()` / `from_dict()` serialization.
- Canonicalization of binary nodes for consistent ordering.
- Program equality and hashing based on canonical form.
- Complexity counting by node count.

### `core/executor.py`
Implements `ProgramExecutor` to execute Defense Programs over prompts.
- `execute(program, prompt)` returns `Outcome`.
- `execute_with_trace(program, prompt)` returns outcome and execution trace.
- Supports evaluation of predicates, classifiers, thresholds, transforms, boolean operators, and if-then-else logic.

### `core/intervention.py`
Defines the `Intervention` model.
- Holds `base_prompt`, ordered `transforms`, optional `metadata`, and generated `id`.
- Computes `final_prompt` lazily via `apply()`.
- Supports `to_dict()` and `from_dict()` serialization using the primitive registry.

### `core/observation.py`
Defines the `Observation` model.
- Includes `intervention_id`, `victim_id`, `campaign_id`, `experiment_id`, `outcome`, `raw_response`, `latency`, `token_usage`, `environment_metadata`, `timestamp`, `metadata`, and `id`.
- Supports `to_dict()` and `from_dict()`.

### `core/hypothesis.py`
Defines the `Hypothesis` model.
- Includes `id`, `statement`, `program`, `belief`, `confidence`, `supporting_observations`, `opposing_observations`, `related_policies`, `provenance`, `metadata`, `created_at`, `updated_at`, and `status`.
- Supports provenance recording and round-trip serialization via `to_dict()` and `from_dict()`.

### `core/grammar.py`
Includes a simple SMT-LIB grammar generator for future synthesis integration.
- `Grammar.to_smtlib()` creates a string stub with all registered primitive names.
- `get_primitive_names()` returns available primitives from the registry.

### `core/utils.py`
Utility helpers for program handling.
- `canonicalize_program(program)`
- `program_equivalence(p1, p2)`
- `complexity(program)`
- `hash_program(program)`

## Testing
Created pytest coverage for each core module under `tests/core`:
- `test_primitive.py`
- `test_program.py`
- `test_executor.py`
- `test_intervention.py`
- `test_observation.py`
- `test_hypothesis.py`
- `test_grammar.py`
- `test_utils.py`

All tests pass: `14 passed`.

## Notes
- The current implementation is designed for extensibility and is ready for future integration with real LLM behavior, program synthesis, and solver-based optimization.
- Primitives are intentionally mocked for now, with classifiers returning randomized scores and transforms implemented as simple utility operations.
