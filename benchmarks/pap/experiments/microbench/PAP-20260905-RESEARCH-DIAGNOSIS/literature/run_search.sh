#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/fei/research/PD/vllm-pap
search_script=/home/fei/.codex/skills/paper-search/scripts/search_papers.py
out_dir="$repo_root/benchmarks/pap/experiments/microbench/PAP-20260905-RESEARCH-DIAGNOSIS/literature"
test -x "$repo_root/.venv/bin/python"
test -r "$search_script"
mkdir -p "$out_dir"

# No project discovery: the search must not resolve or build serving dependencies.
uv run --no-project --isolated --python "$repo_root/.venv/bin/python" \
  --with requests --with openreview-py "$search_script" \
  --query 'attention feed forward disaggregation LLM inference' \
  --start-year 2024 --end-year 2026 --max-papers 5 \
  --sources semantic_scholar open_alex arxiv openreview crossref dblp \
  > "$out_dir/paper-search-stdout.txt" \
  2> "$out_dir/paper-search-stderr.txt"
