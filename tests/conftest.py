"""Shared pytest fixtures, hooks, and markers for the HARMONY-X test suite."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from dotenv import load_dotenv


# ── Auto-load .env from project root ──────────────────────────────────────
_env_loaded = load_dotenv(Path(__file__).resolve().parent.parent / ".env")
if _env_loaded:
    # CVC5_PATH may have been set by .env
    _p = os.environ.get("CVC5_PATH")
    if _p:
        os.environ["CVC5_PATH"] = str(Path(_p).resolve())


def cvc5_binary_path() -> str:
    """Resolve the CVC5 binary path.

    Resolution order (same as ``synthesis.cvc5_synthesizer._default_cvc5_path``):
    1. ``CVC5_PATH`` environment variable.
    2. ``shutil.which("cvc5")`` on system ``PATH``.
    3. Fallback to ``"cvc5"``.
    """
    env_path = os.environ.get("CVC5_PATH")
    if env_path:
        return env_path
    resolved = shutil.which("cvc5")
    if resolved:
        return resolved
    return "cvc5"


def cvc5_available() -> bool:
    """Check whether the CVC5 binary is reachable and functional."""
    binary = cvc5_binary_path()
    if not shutil.which(binary):
        return False
    try:
        r = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=5,
        )
        return r.returncode == 0
    except Exception:
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "skipif_cvc5_missing: skip the test when CVC5 binary is not available",
    )


@pytest.fixture(scope="session")
def _cvc5_check() -> bool:
    """Session-scoped helper: check once whether CVC5 is available."""
    return cvc5_available()


def _skipif_no_cvc5() -> bool:
    available = cvc5_available()
    if not available:
        return True
    return False
