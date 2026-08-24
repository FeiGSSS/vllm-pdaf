"""Run AIPerf 0.11 with local tokenizer paths supported in workers."""

from pathlib import Path

from aiperf.common.tokenizer import Tokenizer

_resolve_local_snapshot = Tokenizer._resolve_local_snapshot.__func__


def _resolve_snapshot(cls: type[Tokenizer], name: str, revision: str) -> str:
    local_path = Path(name).expanduser()
    if local_path.is_dir():
        return str(local_path.resolve())
    return _resolve_local_snapshot(cls, name, revision)


Tokenizer._resolve_local_snapshot = classmethod(_resolve_snapshot)


if __name__ == "__main__":
    from aiperf.cli import app

    app()
