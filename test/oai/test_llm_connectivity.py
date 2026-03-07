#!/usr/bin/env python3
"""Test LLM connectivity for multiple models in parallel."""

import concurrent.futures
import os

os.environ.setdefault("ANTHROPIC_API_KEY", os.getenv("ANTHROPIC_API_KEY", ""))

import litellm

litellm.suppress_debug_info = True
from litellm import completion

TIMEOUT = 30

MODELS = [
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-haiku-4-5-20251001",
]


def test_model(model: str) -> tuple:
    try:
        resp = completion(
            model=model,
            messages=[{"role": "user", "content": "Who is the president of the United States?"}],
            drop_params=True,
            timeout=TIMEOUT,
        )
        return (model, True, resp.choices[0].message.content)
    except Exception as e:
        return (model, False, str(e))


if __name__ == "__main__":
    print(f"Testing {len(MODELS)} model(s)...\n")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS)) as ex:
        for model, ok, msg in ex.map(test_model, MODELS):
            status = "OK" if ok else "FAIL"
            print(f"[{status}] {model}: {msg}")
