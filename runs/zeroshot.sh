#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd)}"
CONDA_ENV_PATH="${CONDA_ENV_PATH:-}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/data/zeroshot}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${PROJECT_ROOT}/.cache/tokenizer}"
OUTPUT_ROOT="${OUTPUT_ROOT:-.}"
RUN_NAME="${RUN_NAME:-zeroshot-owt-$(date +%Y%m%d)}"
RUN_DIR="${OUTPUT_ROOT}/outputs/${RUN_NAME}"
LOG_ROOT="${LOG_ROOT:-${RUN_DIR}/logs}"
CACHE_DIR="${CACHE_DIR:-${RUN_DIR}/cache}"
CKPT="${CKPT:?Set CKPT to an OWT-pretrained checkpoint}"
DATASETS="${DATASETS:-ptb wikitext103 lm1b lambada ag_news pubmed arxiv}"
BLOCK_SIZE="${BLOCK_SIZE:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NODE_COUNT="${NODE_COUNT:-2}"
NODE_RANK="${NODE_RANK:-0}"
PROC_PER_NODE="${PROC_PER_NODE:-8}"
NUM_SHARDS="${NUM_SHARDS:-$((NODE_COUNT * PROC_PER_NODE))}"
JOB_ID="${JOB_ID:-${RUN_NAME}}"

if [[ -n "${CONDA_ENV_PATH}" ]]; then
    export PATH="${CONDA_ENV_PATH}/bin:${PATH}"
fi
PYTHON_BIN="${PYTHON_BIN:-python}"
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

command -v "${PYTHON_BIN}" >/dev/null
test -f "${PROJECT_ROOT}/scripts/zeroshot.py"
test -f "${PROJECT_ROOT}/scripts/zeroshot_datasets.py"
test -f "${CKPT}"
test -d "${DATA_ROOT}"
test -f "${TOKENIZER_DIR}/tokenizer.pkl"
test -f "${TOKENIZER_DIR}/token_bytes.pt"

if [[ ! "${NODE_RANK}" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NODE_COUNT )); then
    echo "invalid NODE_RANK=${NODE_RANK} for NODE_COUNT=${NODE_COUNT}" >&2
    exit 2
fi
if (( PROC_PER_NODE != 8 || NUM_SHARDS != NODE_COUNT * PROC_PER_NODE )); then
    echo "expected ${NODE_COUNT} nodes x 8 GPUs and ${NODE_COUNT}*8 shards" >&2
    exit 2
fi

cd "${PROJECT_ROOT}"
mkdir -p "${RUN_DIR}" "${LOG_ROOT}" "${CACHE_DIR}"
exec > >(tee -a "${LOG_ROOT}/zeroshot_node${NODE_RANK}.log") 2>&1

SLIM_CHECKPOINT="${CACHE_DIR}/esm_checkpoint_slim.ckpt"
CHECKPOINT_READY="${RUN_DIR}/.zeroshot_checkpoint_ready_${JOB_ID}"
CHECKPOINT_FAILED="${RUN_DIR}/.zeroshot_checkpoint_failed_${JOB_ID}"
FINAL_READY="${RUN_DIR}/.zeroshot_done_${JOB_ID}"
FINAL_FAILED="${RUN_DIR}/.zeroshot_failed_${JOB_ID}"

if (( NODE_RANK == 0 )); then
    if "${PYTHON_BIN}" -m scripts.zeroshot \
        --prepare-checkpoint \
        --checkpoint "${CKPT}" \
        --slim-checkpoint "${SLIM_CHECKPOINT}" \
        --output-dir "${RUN_DIR}" \
        --cache-dir "${CACHE_DIR}"; then
        touch "${CHECKPOINT_READY}"
    else
        touch "${CHECKPOINT_FAILED}"
        exit 1
    fi
fi
while [[ ! -f "${CHECKPOINT_READY}" && ! -f "${CHECKPOINT_FAILED}" ]]; do sleep 2; done
[[ ! -f "${CHECKPOINT_FAILED}" ]]

read -r -a DATASET_LIST <<< "${DATASETS}"
for dataset in "${DATASET_LIST[@]}"; do
    prepared="${RUN_DIR}/.zeroshot_prepare_${JOB_ID}_${dataset}"
    prepare_failed="${prepared}.failed"
    if (( NODE_RANK == 0 )); then
        if "${PYTHON_BIN}" -m scripts.zeroshot \
            --dataset "${dataset}" \
            --data-root "${DATA_ROOT}" \
            --tokenizer-dir "${TOKENIZER_DIR}" \
            --output-dir "${RUN_DIR}" \
            --cache-dir "${CACHE_DIR}" \
            --block-size "${BLOCK_SIZE}" \
            --prepare-only; then
            touch "${prepared}"
        else
            touch "${prepare_failed}"
        fi
    fi
    while [[ ! -f "${prepared}" && ! -f "${prepare_failed}" ]]; do sleep 2; done
    [[ ! -f "${prepare_failed}" ]]

    pids=()
    shard_failed=0
    for local_gpu in $(seq 0 $((PROC_PER_NODE - 1))); do
        shard_index=$((NODE_RANK * PROC_PER_NODE + local_gpu))
        log_file="${LOG_ROOT}/${dataset}_shard${shard_index}.log"
        CUDA_VISIBLE_DEVICES="${local_gpu}" "${PYTHON_BIN}" -m scripts.zeroshot \
            --checkpoint "${SLIM_CHECKPOINT}" \
            --source-checkpoint "${CKPT}" \
            --dataset "${dataset}" \
            --data-root "${DATA_ROOT}" \
            --tokenizer-dir "${TOKENIZER_DIR}" \
            --output-dir "${RUN_DIR}" \
            --cache-dir "${CACHE_DIR}" \
            --block-size "${BLOCK_SIZE}" \
            --batch-size "${BATCH_SIZE}" \
            --device cuda \
            --shard-index "${shard_index}" \
            --num-shards "${NUM_SHARDS}" >"${log_file}" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "${pid}" || shard_failed=1
    done
    if (( shard_failed != 0 )); then
        touch "${RUN_DIR}/.zeroshot_dataset_failed_${JOB_ID}_${dataset}"
        exit 1
    fi
done

if (( NODE_RANK == 0 )); then
    if {
        for dataset in "${DATASET_LIST[@]}"; do
            "${PYTHON_BIN}" -m scripts.zeroshot \
                --dataset "${dataset}" \
                --output-dir "${RUN_DIR}" \
                --cache-dir "${CACHE_DIR}" \
                --merge-shards
        done
        "${PYTHON_BIN}" -m scripts.zeroshot \
            --output-dir "${RUN_DIR}" \
            --cache-dir "${CACHE_DIR}" \
            --aggregate
    }; then
        touch "${FINAL_READY}"
    else
        touch "${FINAL_FAILED}"
        exit 1
    fi
else
    while [[ ! -f "${FINAL_READY}" && ! -f "${FINAL_FAILED}" ]]; do sleep 2; done
    [[ ! -f "${FINAL_FAILED}" ]]
fi

echo "zero-shot evaluation completed: ${RUN_DIR}"
