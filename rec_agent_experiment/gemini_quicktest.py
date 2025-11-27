#!/usr/bin/env python3
"""
Minimal Gemini quick test.
Requires:
    pip install google-generativeai
"""

import sys
import google.generativeai as genai
from dotenv import load_dotenv
import os

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
# ENV_PATH = os.path.join(ROOT_DIR, ".env")

load_dotenv()

GEMINI_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_KEY:
    raise ValueError("Missing GEMINI_API_KEY in .env file")


def run_test_gemini(
    api_key: str,
    model: str = "gemini-2.5-flash",
    prompt: str = "Say hello in one sentence",
    temperature: float = 0.2,
    max_tokens: int = 128,
):
    genai.configure(api_key=api_key)
    client = genai.GenerativeModel(model)

    response = client.generate_content(
        prompt,
        generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
    )
    print(response.text)


if __name__ == "__main__":
    API_KEY = GEMINI_KEY

    if not API_KEY and len(sys.argv) > 1:
        API_KEY = sys.argv[1]

    if not API_KEY:
        raise SystemExit("缺少 API Key，请在脚本中设置 API_KEY 或通过命令行传入")

    run_test_gemini(api_key=API_KEY)
