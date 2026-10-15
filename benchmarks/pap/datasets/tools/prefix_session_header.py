#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefix each session's first turn with a unique session line.

The salt-free L07 regeneration shares about 90% of every first-turn document
across sessions. A distinct first line makes every session's first KV block,
and therefore every chained block hash, unique without ``cache_salt``.
"""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = []
    for raw in args.source.read_text().splitlines():
        session = json.loads(raw)
        first = session["turns"][0]
        number = session["session_id"].rsplit("-", 1)[1]
        first["text"] = f"Session {number} ({session['session_id']}).\n{first['text']}"
        lines.append(json.dumps(session, ensure_ascii=False, separators=(",", ":")))
    data = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(data)
    print(hashlib.sha256(data.encode()).hexdigest())


if __name__ == "__main__":
    main()
