# ESM

ESM is a compact energy-based language-model repository. It contains one
paper-aligned model implementation and small entry points for pretraining,
supervised fine-tuning, evaluation, QA, zero-shot evaluation, and generation.

## Install

From the repository root:

```bash
uv sync --extra gpu
```

Use `--extra cpu` on a CPU-only machine. Add `--group dev` for the test
dependencies:

```bash
uv sync --extra cpu --group dev
```

The PyTorch CPU and CUDA indexes are configured in `pyproject.toml`. Choose
one extra per environment.

## Run

The shell launchers are deliberately thin. They always run from the project
root, so relative paths are stable in local shells and distributed jobs.

```bash
# Pretraining: training length comes from TARGET_TOTAL_TOKENS.
TARGET_TOTAL_TOKENS=100000000 bash runs/train.sh

# Supervised fine-tuning from an explicit checkpoint.
bash runs/sft.sh \
  --execution_mode finetune \
  --dataset_name esm_sft \
  --finetuning_model_ckpt /path/to/pretrain.ckpt

# Position-wise BPB evaluation.
bash runs/eval.sh \
  --checkpoint /path/to/model.ckpt \
  --dataset dclm \
  --data-dir /path/to/data \
  --output_root . \
  --run_name eval-dclm

# QA evaluation.
bash runs/qa.sh \
  --checkpoint /path/to/model.ckpt \
  --eval-bundle /path/to/eval_bundle \
  --output_root . \
  --run_name qa-dclm

# Interactive generation.
bash runs/chat.sh --checkpoint /path/to/model.ckpt

# Zero-shot evaluation uses an OWT-pretrained checkpoint and prepared data.
CKPT=/path/to/owt-model.ckpt DATA_ROOT=/path/to/zeroshot-data \
  bash runs/zeroshot.sh
```

Training reads `configs/train.yaml`; evaluation reads `configs/eval.yaml`.
Explicit command-line arguments take precedence over YAML values. Training
length is controlled by `TARGET_TOTAL_TOKENS`. The effective gradient
accumulation is derived from `global_batch_size`, `device_batch_size`, and the
distributed world size, so `max_steps` and `accumulate_grad_batches` are not
independent training controls.

Every run uses the same output layout:

```text
outputs/<run-name>/
├── checkpoints/
├── logs/
│   └── wandb/
└── results/
```

Data and tokenizer assets are not included in the repository. Pass their
locations through the configuration or command line. Supported pretraining
and BPB datasets include DCLM, FineWeb, ClimbMix, and OWT.

## Standalone checkpoint loading

`esm/modeling_esm.py` contains the model classes and the checkpoint loader.
To publish a model repository, upload the checkpoint, tokenizer assets, and
this single Python file:

```text
model-repository/
├── modeling_esm.py
├── model.ckpt
└── tokenizer/
    ├── tokenizer.pkl
    └── token_bytes.pt
```

Install PyTorch and `tiktoken`, copy `modeling_esm.py` beside the checkpoint,
and run:

```python
import torch
from modeling_esm import load_checkpoint

model, tokenizer, hparams, device = load_checkpoint(
    "model.ckpt",
    tokenizer_path="tokenizer",
)

input_ids = torch.tensor(
    [tokenizer.encode("hello", append=tokenizer.get_bos_token_id())],
    device=device,
)
logits = model(input_ids)
print(logits.shape)
```

If the checkpoint and `tokenizer/` directory are siblings, omit
`tokenizer_path`. `token_bytes.pt` is needed for BPB evaluation; logits-only
inference only needs `tokenizer.pkl`. The loader accepts the Lightning
checkpoint format and reconstructs the architecture from its saved
hyperparameters. It is not a Transformers `PreTrainedModel`.

## Repository layout

```text
configs/      Reproducible training and evaluation defaults
esm/          Model, data, optimizer, tokenizer, and checkpoint code
scripts/      Python entry points
runs/         Local shell launchers
tasks/        QA and supervised-task definitions
tests/        Lightweight unit tests
```

Cluster-specific rjob submitters, private data, checkpoints, caches, logs,
and generated outputs are kept outside the public source tree or ignored by
`.gitignore`.

## Development

```bash
uv sync --extra cpu --group dev
pytest -q
python -m scripts.train --help
python -m scripts.eval --help
python -m scripts.qa --help
python -m scripts.chat --help
```

The public tree is intended to stay small: the model, its data interfaces,
the runnable entry points, and lightweight tests are the source of truth.
