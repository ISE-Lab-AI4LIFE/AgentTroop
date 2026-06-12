"""Shared pytest fixtures, hooks, and markers for the HARMONY-X test suite."""

from pathlib import Path

import pytest
from dotenv import load_dotenv


# ── Auto-load .env from project root ──────────────────────────────────────
_env_loaded = load_dotenv(Path(__file__).resolve().parent.parent / ".env")
