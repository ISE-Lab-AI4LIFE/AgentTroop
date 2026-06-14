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

    # Built-in mappings: name -> (package_or_module_path, class_name, _)
    builtins: Dict[str, tuple] = {
        "toy": ("experiments.toy_model", "ToyVictim", None),
        "llama31_8b": ("victim.ollama", "OllamaVictim", None),
        "llama3_1_8b": ("victim.ollama", "OllamaVictim", None),
        "gemma": ("llm.llm_client", "GemmaVictim", None),
        "openrouter": ("victim.openrouter", "OpenRouterVictim", None),
    }

    if name in _VICTIM_REGISTRY:
        cls = _VICTIM_REGISTRY[name]
        logger.info("Creating victim '%s' from registry", name)
        return cls(**kwargs)

    if name in builtins:
        module_path, class_name, _ = builtins[name]
        try:
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
    from victim.ollama import OllamaVictim
    register("llama31_8b", OllamaVictim)
    register("llama3_1_8b", OllamaVictim)
except Exception:
    pass
