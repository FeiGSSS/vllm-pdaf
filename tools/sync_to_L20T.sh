#!/usr/bin/env bash
set -euo pipefail

# One-command working-tree + Python environment sync to the peer host.
#
# This mirrors the local checkout and .venv to the peer host. It is a working
# tree sync, not a git protocol replacement: tracked changes should still be
# committed and fetched when the peer host has git access to the same origin.
#
# Usage:
#   bash tools/sync_to_L20T.sh
#
# Optional overrides:
#   SYNC_HOST     target SSH host (default: L20T)
#   SYNC_VENV     set to 0 to skip the Python environment sync
#   SYNC_SOURCE   set to 0 to skip the code working-tree sync

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SYNC_HOST="${SYNC_HOST:-L20T}"
REMOTE_ROOT="${SYNC_REMOTE_ROOT:-/home/fei/research/PD/vllm-pap}"
SYNC_VENV="${SYNC_VENV:-1}"
SYNC_SOURCE="${SYNC_SOURCE:-1}"

echo "Syncing local ${ROOT_DIR} -> ${SYNC_HOST}:${REMOTE_ROOT}"

if [[ "${SYNC_VENV}" == "1" ]]; then
  echo "[1/2] Syncing .venv"
  rsync -a --delete --partial \
    "${ROOT_DIR}/.venv/" \
    "${SYNC_HOST}:${REMOTE_ROOT}/.venv/"
fi

if [[ "${SYNC_SOURCE}" == "1" ]]; then
  echo "[2/2] Syncing code working tree"
  rsync -a --delete --partial \
    --exclude='.git/' \
    --exclude='.venv/' \
    --exclude='.venv-aiperf/' \
    --exclude='.venv-dynamo/' \
    --exclude='.local/' \
    --exclude='.deps/' \
    --exclude='.claude/' \
    --exclude='.codex/' \
    --exclude='.agents/' \
    --exclude='.remember/' \
    --exclude='.superpowers/' \
    --exclude='.worktrees/' \
    --exclude='.mypy_cache/' \
    --exclude='.pytest_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.gitignore' \
    "${ROOT_DIR}/" \
    "${SYNC_HOST}:${REMOTE_ROOT}/"
fi

echo "Sync complete."
