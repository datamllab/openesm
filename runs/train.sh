#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/train.yaml}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

read -r -a EXTRA_ARGS <<< "${TRAIN_ARGS:-}"
TRAIN_ARGS=(--config "${CONFIG_FILE}" "${EXTRA_ARGS[@]}" "$@")

# Export only what the caller explicitly set. Defaulting to 1 here would shadow
# the YAML values that esm/config.py falls back to.
if [[ -n "${NODE_COUNT:-}" ]]; then export NODE_COUNT; fi
if [[ -n "${PROC_PER_NODE:-}" ]]; then export PROC_PER_NODE; fi

if [[ "${DISTRIBUTED_JOB:-false}" == "true" || "${NODE_COUNT:-1}" -gt 1 || "${PROC_PER_NODE:-1}" -gt 1 ]]; then
    if [[ -n "${MASTER_ADDR:-}" && -n "${NODE_RANK:-}" ]]; then
        exec torchrun \
            --nnodes="${NODE_COUNT:-1}" \
            --node_rank="${NODE_RANK}" \
            --nproc_per_node="${PROC_PER_NODE:-1}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT:-29500}" \
            --module scripts.train \
            "${TRAIN_ARGS[@]}"
    fi
    exec torchrun --standalone --nproc_per_node="${PROC_PER_NODE:-1}" \
        --module scripts.train "${TRAIN_ARGS[@]}"
fi

exec python -m scripts.train "${TRAIN_ARGS[@]}"
