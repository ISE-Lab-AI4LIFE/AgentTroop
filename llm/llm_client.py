import os
from typing import Any, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables from a local .env file if present
load_dotenv()


class LLMClient:
	"""Light wrapper around Google GenAI client providing a stable `generate()` method.

	The wrapper tries a few common client call patterns to be compatible with
	different `google-genai` versions (e.g., `client.generate(...)` or
	`client.responses.create(...)`).
	"""

	def __init__(self, api_key: Optional[str] = None, model_name: str = "gemma-4-31b-it"):
		api_key = api_key or os.environ.get("GEMMA_API_KEY") or os.environ.get("GENAI_API_KEY")
		if not api_key:
			raise RuntimeError(
				"GEMMA_API_KEY environment variable is not set. Set GEMMA_API_KEY (or GENAI_API_KEY) to your GenAI API key."
			)

		self.client = genai.Client(api_key=api_key)
		self.model = model_name

	def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0, **kwargs) -> str:
		"""Generate text from the model.

		Tries multiple client call patterns and returns the final text output.
		"""
		# 1) Try `client.generate(...)` (older/newer variants)
		try:
			if hasattr(self.client, "generate"):
				resp = self.client.generate(model=self.model, prompt=prompt, max_output_tokens=max_tokens, temperature=temperature, **kwargs)
				# Try to extract text sensibly
				if hasattr(resp, "text"):
					return resp.text
				# Some versions return dict-like
				if isinstance(resp, dict):
					return resp.get("output_text") or resp.get("text") or str(resp)

		except Exception:
			pass

		# 2) Try `client.responses.create(...)` pattern
		try:
			if hasattr(self.client, "responses") and hasattr(self.client.responses, "create"):
				resp = self.client.responses.create(model=self.model, input=prompt, max_output_tokens=max_tokens, temperature=temperature, **kwargs)
				# Typical structure: resp.output[0].content[0].text OR resp.output_text
				if hasattr(resp, "output_text"):
					return resp.output_text
				# Try nested extraction
				out = getattr(resp, "output", None)
				if out:
					try:
						# handle sequences of content
						first = out[0]
						content = getattr(first, "content", None)
						if content and len(content) > 0 and hasattr(content[0], "text"):
							return content[0].text
					except Exception:
						pass
				return str(resp)

		except Exception:
			pass

		# 3) Fallback: return empty or raise
		raise RuntimeError("Unable to call GenAI client: unsupported client API surface in installed google-genai package.")


def get_default_client() -> LLMClient:
	"""Convenience helper to get a module-level LLMClient using env vars."""
	return LLMClient()


if __name__ == "__main__":
	# Quick smoke test when run directly
	c = get_default_client()
	print("Client ready — model:", c.model)