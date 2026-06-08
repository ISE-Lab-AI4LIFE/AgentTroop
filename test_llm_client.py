"""Test script for LLMClient from llm/llm_client.py.

Usage:
    python test_llm_client.py
    GEMMA_API_KEY=... python test_llm_client.py  --no-env
"""

"""Test script for OpenRouterClient from llm/llm_client.py.

Usage:
    python test_llm_client.py
    python test_llm_client.py --no-env   # skip tests that need API calls
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm.llm_client import OpenRouterClient, get_default_client


def test_get_default_client():
    print("=== test_get_default_client ===")
    client = get_default_client()
    print(f"  Model: {client.model}")
    print("  OK")


def test_generate_basic():
    print("\n=== test_generate_basic ===")
    client = get_default_client()
    resp = client.generate("Say hello in one word.")
    print(f"  Response: {resp!r}")
    assert isinstance(resp, str) and len(resp) > 0
    print("  OK")


def test_generate_with_params():
    print("\n=== test_generate_with_params ===")
    client = get_default_client()
    resp = client.generate(
        "Write exactly two sentences about AI safety.",
        max_tokens=256,
        temperature=0.7,
    )
    print(f"  Response: {resp!r}")
    assert isinstance(resp, str) and len(resp) > 0
    print("  OK")


def test_generate_long_output():
    print("\n=== test_generate_long_output ===")
    client = get_default_client()
    resp = client.generate(
        "Write a paragraph about machine learning.",
        max_tokens=1024,
    )
    word_count = len(resp.split())
    print(f"  Response ({word_count} words): {resp[:120]}...")
    assert word_count > 5
    print("  OK")


def test_generate_empty_prompt():
    print("\n=== test_generate_empty_prompt ===")
    client = get_default_client()
    try:
        resp = client.generate("")
        print(f"  Response: {resp!r}")
    except Exception as e:
        print(f"  Got error (acceptable for empty prompt): {e}")
    print("  OK")


def test_custom_model():
    print("\n=== test_custom_model ===")
    client = OpenRouterClient(model="gemma-4-31b-it")
    resp = client.generate("Reply with the word 'ok'.")
    print(f"  Response: {resp!r}")
    print("  OK")


def test_no_api_key():
    print("\n=== test_no_api_key ===")
    original = os.environ.pop("OPENROUTER_API_KEY", None)
    try:
        OpenRouterClient()
        print("  FAIL: expected RuntimeError")
    except RuntimeError as e:
        print(f"  Got expected error: {e}")
        print("  OK")
    finally:
        if original is not None:
            os.environ["OPENROUTER_API_KEY"] = original


def _needs_api_key(func):
    """Decorator: wrap test so it catches API quota errors gracefully."""
    name = func.__name__
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e) or "quota" in str(e).lower():
                print(f"  SKIP ({name}): quota exhausted (429)")
            else:
                raise
    return wrapper


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test OpenRouterClient")
    parser.add_argument("--no-env", action="store_true",
                        help="Skip tests that require GEMMA_API_KEY")
    args = parser.parse_args()

    if args.no_env:
        print("Skipping tests that require GEMMA_API_KEY")
    else:
        if "GEMMA_API_KEY" not in os.environ and "GENAI_API_KEY" not in os.environ:
            print("WARNING: Neither GEMMA_API_KEY nor GENAI_API_KEY is set.")
            print("Tests requiring a real API key will fail.")
            print("Set the environment variable or pass --no-env to skip.")
            print()

        try:
            test_get_default_client()
        except Exception as e:
            print(f"  FAIL (get_default_client): {e}")

        for fn in [test_generate_basic, test_generate_with_params,
                   test_generate_long_output, test_generate_empty_prompt,
                   test_custom_model]:
            name = fn.__name__
            try:
                fn()
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e):
                    print(f"  SKIP ({name}): quota exhausted (429)")
                else:
                    print(f"  FAIL ({name}): {e}")

    try:
        test_no_api_key()
    except Exception as e:
        print(f"  FAIL (no_api_key): {e}")

    print("\nDone.")
