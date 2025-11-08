#!/usr/bin/env python3
"""
Minimal Gemini test using google-genai (official client).

- Does NOT require environment variables; pass API key as a string.
  Option A: edit API_KEY below.
  Option B: pass API key as first CLI arg:  python rec_agent_experiment/gemini_quicktest.py <API_KEY>

Prereq:
  pip install google-genai

Docs:
  https://ai.google.dev/gemini-api/docs/get-started/python
"""

from typing import Optional
import sys

try:
    from google import genai
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "google-genai is not installed. Install with: pip install google-genai\n"
        f"Import error: {e}"
    )


def run_test(
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt: str = "Explain how AI works in a few words",
) -> None:
    """Call Gemini with a simple prompt and print the text response."""
    client = genai.Client(api_key=api_key)
    resp = client.models.generate_content(model=model, contents=prompt)
    print(resp.text)


if __name__ == "__main__":
    # Option A: put your API key string here (recommended for a quick test, do NOT commit real keys)
    API_KEY: Optional[str] = ""  # e.g., "AIza..."  <-- fill here if you want

    # Option B: provide API key via first CLI arg
    if not API_KEY and len(sys.argv) > 1:
        API_KEY = sys.argv[1]

    if not API_KEY:
        raise SystemExit(
            "Missing API key. Either set API_KEY in this file or pass it as the first CLI argument.\n"
            "Usage: python rec_agent_experiment/gemini_quicktest.py <API_KEY>"
        )

    run_test(api_key="")
