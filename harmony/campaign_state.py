"""Save/load all learned campaign state to disk for reuse during inference.

All dumped objects survive process restart so that the ASR evaluation
can reconstruct the full victim model learned during the probing phase.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Technique stats ──────────────────────────────────────────────────────────

def save_technique_stats(path: str, stats: Dict[str, Dict[str, Any]]) -> None:
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)


def load_technique_stats(path: str) -> Dict[str, Dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Surrogate Policy Model ──────────────────────────────────────────────────

def save_surrogate(path_stem: str, surrogate: Any) -> None:
    """Save surrogate policy model state, sklearn model (pickle), and scaler.

    Three files are created::

        {path_stem}_state.json      — MLP state dict + metadata
        {path_stem}_model.pkl       — sklearn model (can be None)
        {path_stem}_scaler.pkl      — StandardScaler (can be None)
    """
    if surrogate is None:
        with open(f"{path_stem}_state.json", "w") as f:
            json.dump({}, f)
        with open(f"{path_stem}_model.pkl", "wb") as f:
            pickle.dump(None, f)
        with open(f"{path_stem}_scaler.pkl", "wb") as f:
            pickle.dump(None, f)
        return
    state = surrogate.state_dict()
    state["n_episodes"] = getattr(surrogate, "_n_episodes", 0)
    state["is_trained"] = getattr(surrogate, "_is_trained", False)
    with open(f"{path_stem}_state.json", "w") as f:
        json.dump(state, f, indent=2)

    model = getattr(surrogate, "_sklearn_model", None)
    with open(f"{path_stem}_model.pkl", "wb") as f:
        pickle.dump(model, f)

    scaler = getattr(surrogate, "_scaler", None)
    with open(f"{path_stem}_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


def load_surrogate(path_stem: str) -> Dict[str, Any]:
    """Load surrogate state into a dict for later reconstruction."""
    state_path = f"{path_stem}_state.json"
    model_path = f"{path_stem}_model.pkl"
    scaler_path = f"{path_stem}_scaler.pkl"

    if not os.path.exists(state_path):
        return {}

    with open(state_path) as f:
        state = json.load(f)

    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            state["_sklearn_model"] = pickle.load(f)
    else:
        state["_sklearn_model"] = None

    if os.path.exists(scaler_path):
        with open(scaler_path, "rb") as f:
            state["_scaler"] = pickle.load(f)
    else:
        state["_scaler"] = None

    return state


def restore_surrogate(surrogate: Any, state: Dict[str, Any]) -> None:
    """Restore a SurrogatePolicyModel from saved state."""
    # Only pass actual MLP parameter keys to load_state_dict
    param_keys = {k for k in surrogate.state_dict()}
    model_state = {k: v for k, v in state.items() if k in param_keys}
    surrogate.load_state_dict(model_state)
    surrogate._sklearn_model = state.get("_sklearn_model")
    surrogate._scaler = state.get("_scaler")
    surrogate._is_trained = state.get("is_trained", False)
    surrogate._n_episodes = state.get("n_episodes", 0)


# ── Version Space ────────────────────────────────────────────────────────────

def save_version_space(path: str, version_space: Any) -> None:
    """Save version-space candidates + program ASTs + posterior."""
    if version_space is None:
        with open(path, "w") as f:
            json.dump({}, f)
        return
    vs_dict = version_space.to_dict()
    # Also include the full program AST for each candidate so that
    # downstream consumers can make predictions without the process memory.
    from inference.version_space import CandidateProgram
    asts = {}
    for c in version_space.candidates:
        try:
            asts[c.program_id] = c.program.to_dict() if hasattr(c.program, "to_dict") else str(c.program)
        except Exception:
            asts[c.program_id] = str(c.program)
    vs_dict["program_asts"] = asts
    with open(path, "w") as f:
        json.dump(vs_dict, f, indent=2, default=str)


def load_version_space(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── SDE Engine ───────────────────────────────────────────────────────────────

def save_sde_engine(path: str, sde_engine: Any) -> None:
    """Save SDE boundary estimators and engine state."""
    if sde_engine is None:
        with open(path, "w") as f:
            json.dump({}, f)
        return
    state = {
        "_round": getattr(sde_engine, "_round", 0),
        "_converged": getattr(sde_engine, "_converged", False),
        "history": getattr(sde_engine, "_history", []),
        "boundary_estimators": {},
    }
    for name, estimator in getattr(sde_engine, "boundary_estimators", {}).items():
        obs = getattr(estimator, "_observations", [])
        direction = getattr(estimator, "_direction", "positive")
        state["boundary_estimators"][name] = {
            "observations": [(float(s), int(o)) for s, o in obs],
            "direction": direction,
            "num_observations": len(obs),
        }
    with open(path, "w") as f:
        json.dump(state, f, indent=2, default=str)


def load_sde_engine(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── Causal Graph cumulative outcomes ────────────────────────────────────────

def save_causal_accumulator(path: str, causal_graph: Any) -> None:
    if causal_graph is None:
        with open(path, "w") as f:
            json.dump({}, f)
        return
    cum = getattr(causal_graph, "_cumulative_outcomes", {})
    with open(path, "w") as f:
        json.dump(cum, f, indent=2, default=str)


def load_causal_accumulator(path: str) -> Dict[str, Dict[str, List[int]]]:
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


# ── High-level API ───────────────────────────────────────────────────────────

def save_campaign_state(
    output_dir: str,
    campaign_id: str,
    version_space: Any,
    surrogate: Any,
    sde_engine: Any,
    causal_graph: Any,
    technique_stats: Dict[str, Dict[str, Any]],
) -> str:
    """Persist all learned components to *output_dir* / *campaign_id*.

    Returns the full path to the campaign state directory.
    """
    state_dir = os.path.join(output_dir, campaign_id)
    os.makedirs(state_dir, exist_ok=True)

    save_version_space(os.path.join(state_dir, "version_space.json"), version_space)
    save_surrogate(os.path.join(state_dir, "surrogate"), surrogate)
    save_sde_engine(os.path.join(state_dir, "sde_state.json"), sde_engine)
    save_causal_accumulator(os.path.join(state_dir, "causal_accumulator.json"), causal_graph)
    save_technique_stats(os.path.join(state_dir, "technique_stats.json"), technique_stats)

    logger.info(
        "Campaign state saved: %s (%d VS candidates, surrogate_trained=%s, "
        "%d SDE estimators, %d causal keys, %d techniques)",
        state_dir,
        version_space.num_candidates if version_space else 0,
        getattr(surrogate, "_is_trained", False),
        len(getattr(sde_engine, "boundary_estimators", {})),
        len(getattr(causal_graph, "_cumulative_outcomes", {})),
        len(technique_stats),
    )
    return state_dir


def load_campaign_state(state_dir: str) -> Dict[str, Any]:
    """Load all persisted campaign knowledge into a flat dict.

    Keys::

        version_space     — dict (candidates, posteriors, program_asts)
        surrogate         — dict (state + _sklearn_model, _scaler)
        sde_engine        — dict (_round, boundary_estimators, history)
        causal_graph      — dict (source_target -> outcomes dict)
        technique_stats   — dict (technique -> stats)
    """
    return {
        "version_space": load_version_space(os.path.join(state_dir, "version_space.json")),
        "surrogate": load_surrogate(os.path.join(state_dir, "surrogate")),
        "sde_engine": load_sde_engine(os.path.join(state_dir, "sde_state.json")),
        "causal_graph": load_causal_accumulator(os.path.join(state_dir, "causal_accumulator.json")),
        "technique_stats": load_technique_stats(os.path.join(state_dir, "technique_stats.json")),
    }
