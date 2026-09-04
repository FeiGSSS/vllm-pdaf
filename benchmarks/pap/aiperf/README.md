# PAP AIPerf adapter

AIPerf is installed from PyPI into `.venv-aiperf`; its source and environment
do not live in this directory. This directory contains only the thin adapter
used by PAP and Dynamo experiment runners.

## Environment

```bash
UV_CACHE_DIR=/tmp/uv-cache-aiperf \
  uv venv --python 3.12 --seed .venv-aiperf
UV_CACHE_DIR=/tmp/uv-cache-aiperf \
  uv pip install --python .venv-aiperf/bin/python \
    -r benchmarks/pap/aiperf/requirements.txt
```

## Files

| File | Purpose |
| --- | --- |
| `requirements.txt` | Pin the shared PyPI AIPerf version |
| `aiperf_compat_entry.py` | Support local tokenizer paths and DCGM metrics |
| `run_profile.sh` | Run one AIPerf profile against an existing endpoint |

Workload files and dataset construction tools live in `../datasets/`.
Experiment-specific QPS, concurrency, duration, topology, and dataset identity
belong in each experiment's `experiment.env`, not in this shared adapter.
