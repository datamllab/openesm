#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_FILE="${CONFIG_FILE:-${REPO_ROOT}/configs/train.yaml}"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
cd "${REPO_ROOT}"

read -r -a EXTRA_ARGS <<< "${SFT_ARGS:-}"
SFT_ARGS=(--config "${CONFIG_FILE}" "${EXTRA_ARGS[@]}" "$@")

NODE_COUNT="${NODE_COUNT:-1}"
PROC_PER_NODE="${PROC_PER_NODE:-1}"
export NODE_COUNT PROC_PER_NODE
if [[ "${DISTRIBUTED_JOB:-false}" == "true" || "${NODE_COUNT}" -gt 1 || "${PROC_PER_NODE}" -gt 1 ]]; then
    if [[ -n "${MASTER_ADDR:-}" && -n "${NODE_RANK:-}" ]]; then
        exec torchrun \
            --nnodes="${NODE_COUNT}" \
            --node_rank="${NODE_RANK}" \
            --nproc_per_node="${PROC_PER_NODE}" \
            --master_addr="${MASTER_ADDR}" \
            --master_port="${MASTER_PORT:-29500}" \
            --module scripts.sft \
            "${SFT_ARGS[@]}"
    fi
    exec torchrun --standalone --nproc_per_node="${PROC_PER_NODE}" \
        --module scripts.sft "${SFT_ARGS[@]}"
fi

exec python -m scripts.sft "${SFT_ARGS[@]}"
