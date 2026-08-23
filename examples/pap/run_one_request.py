# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Send one OpenAI completions request and optionally persist the response."""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one PAP completion request")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--model", default="/data/ssd1/llm-models/Qwen3-8B")
    parser.add_argument("--prompt", default="Briefly explain what PAP does.")
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conversation-id", default="")
    parser.add_argument("--output", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = {
        "model": args.model,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stream": False,
    }
    if args.conversation_id:
        payload["conversation_id"] = args.conversation_id
    resp = httpx.post(
        f"http://{args.host}:{args.port}/v1/completions",
        json=payload,
        timeout=None,
    )
    resp.raise_for_status()
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(resp.text + "\n")
    print(resp.text)


if __name__ == "__main__":
    main()
