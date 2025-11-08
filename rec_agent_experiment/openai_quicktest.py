#!/usr/bin/env python3
"""
Minimal OpenAI GPT quick test.

- 不依赖环境变量，直接在脚本里填入字符串 API Key；或运行时作为参数传入。
- 需要先安装 openai 官方 SDK:  pip install openai
"""

from typing import Optional
import sys

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "openai SDK 未安装，请先执行: pip install openai\n" f"Import error: {exc}"
    )


def run_test(
    api_key: str,
    model: str = "gpt-4o-mini",
    prompt: str = "Say hello in one sentence",
    temperature: float = 0.2,
    max_tokens: int = 128,
):
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    text = response.choices[0].message.content
    print(text)


if __name__ == "__main__":
    # Option A: 直接在这里填写 key（注意避免提交到版本库）
    API_KEY: Optional[str] = ""  # e.g. "sk-..."

    # Option B: 通过命令行参数传 key
    if not API_KEY and len(sys.argv) > 1:
        API_KEY = sys.argv[1]

    if not API_KEY:
        raise SystemExit(
            "缺少 API Key。请在脚本中设置 API_KEY，或运行: \n"
            "  python rec_agent_experiment/openai_quicktest.py <API_KEY>"
        )

    run_test(api_key=API_KEY)
