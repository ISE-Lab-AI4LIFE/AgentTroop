"""VictimFactory — unified entry point for creating victims by name.

Supports::

    --victim toy              (default, ground-truth known)
    --victim llama31_8b       (Ollama-hosted Llama 3.1 8B)
    --victim gemma            (Google Gemma via API)
    --victim openrouter       (Any model via OpenRouter API)
    --victim custom:<path>    (Custom victim script)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from adapters.base_victim import BaseVictim

logger = logging.getLogger(__name__)


_VICTIM_REGISTRY: Dict[str, Any] = {}


def register(name: str, cls: Any) -> None:
    """Register a victim class by name."""
    _VICTIM_REGISTRY[name] = cls
    logger.debug("Registered victim '%s': %s", name, cls.__name__)


def create(
    victim_name: str = "toy",
    **kwargs: Any,
) -> BaseVictim:
    """Create a victim instance by name.

    Parameters
    ----------
    victim_name : str
        One of: ``"toy"``, ``"llama31_8b"``, ``"gemma"``, ``"openrouter"``,
        or ``"custom:<module_path>"``.
    **kwargs
        Passed to the victim constructor.

    Returns
    -------
    BaseVictim
    """
    name = victim_name.lower().replace("-", "_").replace(":", "_")

    # Built-in mappings: name -> (package_or_module_path, class_name, file_path_override)
    builtins: Dict[str, tuple] = {
        "toy": ("experiments.toy_model", "ToyVictim", None),
        "llama31_8b": (None, "OllamaVictim", None),
        "llama3.1:8b": (None, "OllamaVictim", None),
        "gemma": ("llm.llm_client", "GemmaVictim", None),
        "openrouter": ("llm.llm_client", "OpenRouterVictim", None),
    }

    if name in _VICTIM_REGISTRY:
        cls = _VICTIM_REGISTRY[name]
        logger.info("Creating victim '%s' from registry", name)
        return cls(**kwargs)

    if name in builtins:
        module_path, class_name, file_override = builtins[name]
        try:
            if name in ("llama31_8b", "llama3.1:8b"):
                # Special handling for non-standard module name
                import importlib.util
                # Try multiple possible paths for the ollama_victim module
                import_paths = [
                    os.path.join(os.path.dirname(__file__), "..", "llama3.1:8b", "ollama_victim.py"),
                    os.path.join(os.path.dirname(__file__), "..", "llama3.1_8b", "ollama_victim.py"),
                ]
                victim_mod = None
                for path in import_paths:
                    resolved = os.path.abspath(path)
                    if os.path.exists(resolved):
                        spec = importlib.util.spec_from_file_location(
                            "ollama_victim", resolved
                        )
                        if spec and spec.loader:
                            victim_mod = importlib.util.module_from_spec(spec)
                            spec.loader.exec_module(victim_mod)
                            break
                if victim_mod is None:
                    raise ImportError("Could not find ollama_victim.py")
                cls = getattr(victim_mod, class_name)
            else:
                import importlib
                mod = importlib.import_module(module_path)
                cls = getattr(mod, class_name)
            logger.info("Creating victim '%s' as %s", name, cls.__name__)
            return cls(**kwargs)
        except (ImportError, AttributeError) as exc:
            logger.warning("Failed to load victim '%s': %s", name, exc)

    # Fallback: toy
    logger.warning("Unknown victim '%s', falling back to toy", name)
    from experiments.toy_model import ToyVictim
    return ToyVictim(**kwargs)


# Register known victims on import
try:
    from experiments.toy_model import ToyVictim
    register("toy", ToyVictim)
except ImportError:
    pass

try:
    from adapters.base_victim import BaseVictim
    # Lazy registration to avoid import errors from the non-standard path
    def _register_ollama():
        import importlib.util
        import os
        paths = [
            os.path.join(os.path.dirname(__file__), "..", "llama3.1:8b", "ollama_victim.py"),
        ]
        for p in paths:
            p = os.path.abspath(p)
            if os.path.exists(p):
                spec = importlib.util.spec_from_file_location("ollama_victim", p)
                if spec and spec.loader:
                    mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(mod)
                    if hasattr(mod, "OllamaVictim"):
                        register("llama31_8b", mod.OllamaVictim)
                        register("llama3.1:8b", mod.OllamaVictim)
                        break
    _register_ollama()
except Exception:
    pass
