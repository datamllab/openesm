<p align="center">
  <img src="assets/openesm-github-title.png" alt="HarborRL" width=800>
</p>

<!-- <p align="center">
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/version-1.0.0-blue.svg" alt="Version"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg" alt="PyTorch 2.4+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="MIT License"></a>
</p> -->




<p align="center">
  <a href="https://arxiv.org/abs/2608.07346"><img src="https://img.shields.io/badge/Paper-b31b1b?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"/></a>
  <a href="https://github.com/datamllab/A2E"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://colab.research.google.com/github/stevewithjobs/AEP/blob/yuchenyue/notebooks/a2e_quickstart.ipynb"><img src="https://img.shields.io/badge/Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white" alt="Colab"/></a>
  <a href="https://huggingface.co/papers/2608.07346"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="HuggingFace"/></a>
</p>

<p align="center">
  <strong>The first repository for training and scaling up energy-based foundation models</strong>
</p>

<p align="center">
  <strong>Keywords: </strong> energy-based models, foundation models, large language models, scaling law
</p>


<p align="center">
  <a href="#-updates">🎉 Updates</a> •
  <a href="#1-quick-start">🚀 Quick Start</a> •
  <a href="#2-scaling-law">📈 Scaling Law</a> •
  <a href="#3-scaling-law">⚙️ Pretrained Checkpoints</a> •
</p>



<p align="center">
  English · <a href="README_zh.md">简体中文</a>
</p>

---

## 🎉 Updates

- **2026-10-07** — ✨✨ Full codebase released.
- **2026-10-07** — 📄 OpenESM preprint posted on [arXiv](https://arxiv.org/abs/2608.07346).


## 🚀 Quick Start

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
