# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sample API-cached Prometheus metrics; never call scheduler utilities."""

import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

output = Path(sys.argv[1])
stop = Path(sys.argv[2])
names = (
    "vllm:request_queue_time_seconds_",
    "vllm:request_prefill_time_seconds_",
    "vllm:request_inference_time_seconds_",
    "vllm:e2e_request_latency_seconds_",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:prefix_cache_hits_total",
)
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sample(pa):
    start = time.time()
    try:
        with opener.open(
            f"http://127.0.0.1:{8100 + pa}/metrics", timeout=1
        ) as response:
            body = response.read().decode()
        lines = [line for line in body.splitlines() if line.startswith(names)]
        return {"time": start, "pa": pa, "elapsed": time.time() - start, "lines": lines}
    except Exception as exc:
        return {"time": start, "pa": pa, "error": type(exc).__name__}


with (
    output.open("x") as stream,
    concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool,
):
    while not stop.exists():
        for row in pool.map(sample, range(7)):
            stream.write(json.dumps(row) + "\n")
        stream.flush()
        time.sleep(0.5)
