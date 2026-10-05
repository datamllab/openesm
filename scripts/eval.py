"""Unified ESM evaluation entry point.

The first-class default is position-wise BPB.  CORE remains available through
the same entry point with ``--task core`` and uses the same checkpoint loader.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import csv
import glob as glob_module
import heapq
import random
import time
from contextlib import contextmanager, nullcontext
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Iterable

from esm.config import REPO_ROOT, parse_with_config, print_resolved_config, resolve_path

torch = None
F = None
mp = None
np = None
yaml = None
tqdm = None
evaluate_example = None
_esm_evaluate_task = None
forward_model = None
render_prompts_mc = None
render_prompts_schema = None
render_prompts_lm = None
batch_sequences_mc = None
batch_sequences_schema = None
batch_sequences_lm = None
stack_sequences = None


def _load_core_runtime() -> None:
    """Load optional torch/CORE dependencies only when evaluation is run."""

    global torch, F, mp, np, yaml, tqdm
    global evaluate_example, _esm_evaluate_task, forward_model
    global render_prompts_mc, render_prompts_schema, render_prompts_lm
    global \
        batch_sequences_mc, \
        batch_sequences_schema, \
        batch_sequences_lm, \
        stack_sequences

    if torch is not None and forward_model is not None:
        return

    from esm.config import bootstrap_assets

    bootstrap_assets("eval")

    import torch as _torch
    import torch.nn.functional as _F
    import torch.multiprocessing as _mp
    import numpy as _np
    import yaml as _yaml
    from tqdm import tqdm as _tqdm
    from esm.core_eval import (
        evaluate_example as _evaluate_example,
        evaluate_task as _evaluate_task,
        forward_model as _forward_model,
        render_prompts_mc as _render_prompts_mc,
        render_prompts_schema as _render_prompts_schema,
        render_prompts_lm as _render_prompts_lm,
        batch_sequences_mc as _batch_sequences_mc,
        batch_sequences_schema as _batch_sequences_schema,
        batch_sequences_lm as _batch_sequences_lm,
        stack_sequences as _stack_sequences,
    )

    torch = _torch
    F = _F
    mp = _mp
    np = _np
    yaml = _yaml
    tqdm = _tqdm
    evaluate_example = _evaluate_example
    _esm_evaluate_task = _evaluate_task
    forward_model = _forward_model
    render_prompts_mc = _render_prompts_mc
    render_prompts_schema = _render_prompts_schema
    render_prompts_lm = _render_prompts_lm
    batch_sequences_mc = _batch_sequences_mc
    batch_sequences_schema = _batch_sequences_schema
    batch_sequences_lm = _batch_sequences_lm
    stack_sequences = _stack_sequences


def _no_grad(function):
    """Lazy equivalent of ``torch.no_grad`` for the importable CLI module."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        _load_core_runtime()
        with torch.no_grad():
            return function(*args, **kwargs)

    return wrapped


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an ESM checkpoint")
    parser.add_argument(
        "--task", choices=("position_bpb", "core"), default="position_bpb"
    )
    parser.add_argument(
        "--checkpoint", default="", help="Checkpoint path; required for evaluation"
    )
    parser.add_argument(
        "--dataset", default="dclm", choices=("fineweb", "climbmix", "dclm", "owt")
    )
    parser.add_argument("--data-dir", dest="data_dir", default=None)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument(
        "--context-length", dest="context_length", type=int, default=2048
    )
    parser.add_argument(
        "--device-batch-size", dest="batch_size_per_device", type=int, default=1
    )
    parser.add_argument(
        "--grad-accum", dest="accumulate_grad_batches", type=int, default=1
    )
    parser.add_argument(
        "--global-batch-size", dest="global_batch_size", type=int, default=0
    )
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument("--virtual-world-size", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        dest="esm_dtype",
        choices=("auto", "bfloat16", "float32"),
        default="auto",
    )
    parser.add_argument("--output-json", default="")
    parser.add_argument("--eval-bundle", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--output_root", default=".")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--log_root", default="")
    parser.add_argument("--max-per-task", type=int, default=-1)
    parser.add_argument("--tokenizer-path", default="")
    parser.add_argument("--gpus", type=int, default=-1)
    parser.add_argument("--task-samples", default="")
    parser.add_argument("--num-trajectory-samples", type=int, default=10)
    parser.add_argument("--history-timing", default="")
    parser.add_argument("--time_embedding", type=str, default="true")
    return parser


def _default_output_json(args: argparse.Namespace) -> str:
    if args.output_json:
        return args.output_json
    result_root = getattr(args, "output_dir", "")
    if result_root:
        return str(Path(result_root) / f"{args.task}.json")
    run_dir = Path(
        getattr(
            args,
            "run_dir",
            Path(getattr(args, "output_root", ".")) / "outputs" / args.run_name,
        )
    )
    return str(run_dir / "results" / f"{args.task}.json")


def _resolve_eval_paths(args: argparse.Namespace) -> argparse.Namespace:
    output_root = resolve_path(
        getattr(args, "output_root", ".") or ".", repo_root=REPO_ROOT
    )
    run_name = (
        getattr(args, "run_name", "") or f"eval-{args.task}-{datetime.now():%Y%m%d}"
    )
    run_dir = output_root / "outputs" / run_name
    log_root = (
        resolve_path(args.log_root, repo_root=REPO_ROOT)
        if args.log_root
        else run_dir / "logs"
    )
    result_dir = (
        resolve_path(args.output_dir, repo_root=REPO_ROOT)
        if args.output_dir
        else run_dir / "results"
    )
    for path in (run_dir, log_root, result_dir):
        path.mkdir(parents=True, exist_ok=True)
    args.output_root = str(output_root)
    args.run_name = run_name
    args.run_dir = str(run_dir)
    args.log_root = str(log_root)
    args.output_dir = str(result_dir)
    if args.output_json:
        args.output_json = str(resolve_path(args.output_json, repo_root=REPO_ROOT))
    return args


def _resolve_data_dir(args: argparse.Namespace) -> Path | None:
    if not args.data_dir:
        return None
    data_dir = Path(args.data_dir)
    dataset_dir = data_dir / args.dataset
    return dataset_dir if dataset_dir.is_dir() else data_dir


def _validate(args: argparse.Namespace) -> None:
    if not args.checkpoint:
        raise SystemExit(
            "--checkpoint is required (the eval.yaml default is intentionally empty)"
        )
    if str(getattr(args, "time_embedding", "true")).lower() not in {
        "true",
        "1",
        "yes",
        "on",
    }:
        raise ValueError("ESM evaluation requires time_embedding=true")


def evaluate(args: argparse.Namespace):
    """Run one evaluation task from an already-resolved namespace.

    Keeping this function separate from argparse makes the evaluator usable by
    notebooks and higher-level launchers while preserving the CLI below.
    """
    _validate(args)

    if args.task == "position_bpb":
        data_dir = _resolve_data_dir(args)
        delegated = argparse.Namespace(
            esm_ckpt=Path(args.checkpoint),
            tokenizer_path=args.tokenizer_path or None,
            dataset=args.dataset,
            data_dir=data_dir,
            split=args.split,
            context_length=args.context_length,
            device_batch_size=args.device_batch_size,
            eval_batches=args.eval_batches,
            virtual_world_size=args.virtual_world_size,
            device=args.device,
            esm_dtype=args.esm_dtype,
            output_json=Path(_default_output_json(args)),
        )
        return evaluate_position_bpb(delegated)

    if not args.eval_bundle:
        raise SystemExit("--eval-bundle is required for --task core")
    output_dir = args.output_dir
    core_argv = [
        "--ckpt-path",
        args.checkpoint,
        "--tokenizer-path",
        args.tokenizer_path or "",
        "--eval-bundle-dir",
        args.eval_bundle,
        "--output-dir",
        output_dir,
        "--max-per-task",
        str(args.max_per_task),
        "--device-batch-size",
        str(args.device_batch_size),
        "--gpus",
        str(args.gpus),
        "--dtype",
        "float32" if args.esm_dtype == "float32" else "bfloat16",
        "--num-trajectory-samples",
        str(args.num_trajectory_samples),
    ]
    if args.task_samples:
        core_argv += ["--task-samples", args.task_samples]
    if args.history_timing:
        core_argv += ["--history-timing", args.history_timing]
    return evaluate_core_cli(core_argv)


def _parse_position_bpb_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-context-position BPB for an ESM checkpoint."
    )
    parser.add_argument(
        "--esm-ckpt",
        type=Path,
        required=True,
        help="Path to the Lightning ESM .ckpt file.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Directory containing tokenizer.pkl and token_bytes.pt.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="fineweb",
        choices=("fineweb", "climbmix", "dclm"),
        help="Base pretraining dataset name.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Optional explicit parquet directory. Overrides the dataset default directory.",
    )
    parser.add_argument("--split", type=str, default="val", choices=("train", "val"))
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--device-batch-size", type=int, default=1)
    parser.add_argument("--eval-batches", type=int, default=50)
    parser.add_argument(
        "--virtual-world-size",
        type=int,
        default=1,
        help=(
            "Simulate this many DDP dataloader ranks in one process. "
            "When >1, --eval-batches is interpreted per virtual rank."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device string, e.g. auto, cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--esm-dtype",
        type=str,
        default="auto",
        choices=("auto", "bfloat16", "float32"),
        help="Dtype used when moving the ESM model to device. auto = bfloat16 on CUDA, float32 otherwise.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        required=True,
        help="Where to write the per-position BPB JSON.",
    )
    return parser.parse_args()


@contextmanager
def virtual_dataloader_rank(rank: int, world_size: int):
    keys = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    old_values = {key: os.environ.get(key) for key in keys}
    if world_size > 1:
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def resolve_esm_dtype(dtype_name: str, device: torch.device) -> torch.dtype:
    if dtype_name == "bfloat16":
        return torch.bfloat16
    if dtype_name == "float32":
        return torch.float32
    return torch.bfloat16 if device.type == "cuda" else torch.float32


def collect_eval_batches(
    tokenizer,
    *,
    batch_size: int,
    context_length: int,
    split: str,
    device: torch.device,
    eval_batches: int,
    dataset: str,
    data_dir: Path | None,
    virtual_world_size: int = 1,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    from esm.dataloader import tokenizing_distributed_data_loader_bos_bestfit

    batches: list[tuple[torch.Tensor, torch.Tensor]] = []
    for rank in range(virtual_world_size):
        if virtual_world_size > 1:
            print(f"Collecting virtual validation rank {rank}/{virtual_world_size - 1}")
        with virtual_dataloader_rank(rank, virtual_world_size):
            loader = tokenizing_distributed_data_loader_bos_bestfit(
                tokenizer,
                batch_size,
                context_length,
                split,
                device=str(device),
                dataset_name=dataset,
                data_dir=str(data_dir) if data_dir is not None else None,
            )
            for _ in range(eval_batches):
                x, y = next(loader)
                batches.append((x.detach().cpu(), y.detach().cpu()))
    return batches


def move_batch(
    batch: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    x, y = batch
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


def _flatten_if_lightning_batch(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def esm_final_step_loss2d(model, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Mirror forward_loss_wrapper's final-step CE path, but keep [B, T]."""
    input_ids = _flatten_if_lightning_batch(x)
    targets = _flatten_if_lightning_batch(y)
    _, _, predicted_hiddens = model(
        input_ids,
        learning=False,
        return_pred_hiddens=True,
    )
    pred_hidden = predicted_hiddens[-1]
    if pred_hidden is None:
        raise RuntimeError("ESM forward returned no post-update hidden state.")
    final_logits = model.tf_head(pred_hidden, model.embeddings(input_ids))
    per_token_ce = F.cross_entropy(
        final_logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
        ignore_index=-1,
        reduction="none",
    )
    return per_token_ce.reshape_as(targets)


def add_position_sums(
    loss2d: torch.Tensor,
    targets: torch.Tensor,
    token_bytes: torch.Tensor,
    nats_by_pos: torch.Tensor,
    bytes_by_pos: torch.Tensor,
    label_logprob_by_pos: torch.Tensor,
    token_count_by_pos: torch.Tensor,
) -> None:
    if loss2d.shape != targets.shape:
        raise ValueError(
            f"loss/target shape mismatch: {tuple(loss2d.shape)} vs {tuple(targets.shape)}"
        )
    if loss2d.shape[1] != nats_by_pos.numel():
        raise ValueError(
            f"Expected context length {nats_by_pos.numel()}, got {loss2d.shape[1]}"
        )

    valid = targets >= 0
    safe_targets = torch.where(valid, targets, torch.zeros_like(targets))
    num_bytes = token_bytes[safe_targets]
    counted = valid & (num_bytes > 0)

    nats_by_pos += (loss2d.detach().to(torch.float64) * counted).sum(dim=0)
    bytes_by_pos += (num_bytes.to(torch.int64) * counted).sum(dim=0)
    label_logprob_by_pos += ((-loss2d.detach()).to(torch.float64) * valid).sum(dim=0)
    token_count_by_pos += valid.to(torch.int64).sum(dim=0)


def compute_position_bpb(
    model_name: str,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    *,
    loss2d_fn,
    device: torch.device,
    context_length: int,
    token_bytes: torch.Tensor,
    autocast_ctx,
) -> dict[str, object]:
    nats_by_pos = torch.zeros(context_length, dtype=torch.float64, device=device)
    bytes_by_pos = torch.zeros(context_length, dtype=torch.int64, device=device)
    label_logprob_by_pos = torch.zeros(
        context_length, dtype=torch.float64, device=device
    )
    token_count_by_pos = torch.zeros(context_length, dtype=torch.int64, device=device)
    num_samples = 0
    num_batches = 0

    for batch in batches:
        x, y = move_batch(batch, device)
        with autocast_ctx:
            loss2d = loss2d_fn(x, y)
        add_position_sums(
            loss2d,
            y,
            token_bytes,
            nats_by_pos,
            bytes_by_pos,
            label_logprob_by_pos,
            token_count_by_pos,
        )
        num_samples += y.shape[0]
        num_batches += 1
        del x, y, loss2d

    nats_cpu = nats_by_pos.cpu()
    bytes_cpu = bytes_by_pos.cpu()
    label_logprob_cpu = label_logprob_by_pos.cpu()
    token_count_cpu = token_count_by_pos.cpu()
    bpb = [
        (float(nats_cpu[i].item()) / (math.log(2) * int(bytes_cpu[i].item())))
        if int(bytes_cpu[i].item()) > 0
        else None
        for i in range(context_length)
    ]
    logit = [
        (float(label_logprob_cpu[i].item()) / int(token_count_cpu[i].item()))
        if int(token_count_cpu[i].item()) > 0
        else None
        for i in range(context_length)
    ]
    total_nats = float(nats_cpu.sum().item())
    total_bytes = int(bytes_cpu.sum().item())
    overall_bpb = total_nats / (math.log(2) * total_bytes) if total_bytes > 0 else None
    valid_position_bpb = [value for value in bpb if value is not None]
    mean_position_bpb = (
        sum(valid_position_bpb) / len(valid_position_bpb)
        if valid_position_bpb
        else None
    )

    return {
        "model": model_name,
        "num_batches": num_batches,
        "num_samples": num_samples,
        "overall_bpb": overall_bpb,
        "mean_position_bpb": mean_position_bpb,
        "position_bpb": bpb,
        "logit": logit,
        "position_nats": [float(v) for v in nats_cpu.tolist()],
        "position_bytes": [int(v) for v in bytes_cpu.tolist()],
        "position_token_count": [int(v) for v in token_count_cpu.tolist()],
    }


def load_esm(args: argparse.Namespace, device: torch.device):
    dtype = resolve_esm_dtype(args.esm_dtype, device)
    from esm.modeling_esm import load_checkpoint

    wrapper, tokenizer, hparams, _ = load_checkpoint(
        str(args.esm_ckpt),
        device=device,
        dtype=dtype,
        tokenizer_path=getattr(args, "tokenizer_path", None),
    )
    wrapper.model.requires_grad_(False)
    wrapper.model.eval()
    return wrapper.model, tokenizer, hparams


def evaluate_position_bpb(args: argparse.Namespace | None = None) -> None:
    args = _parse_position_bpb_args() if args is None else args
    global torch, F
    import torch as _torch
    import torch.nn.functional as _F

    torch = _torch
    F = _F

    if args.eval_batches <= 0:
        raise ValueError("--eval-batches must be positive.")
    if args.device_batch_size <= 0:
        raise ValueError("--device-batch-size must be positive.")
    if args.virtual_world_size <= 0:
        raise ValueError("--virtual-world-size must be positive.")
    device_arg = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_arg == "auto":
        device_arg = "cpu"
    device = torch.device(device_arg)
    autocast_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device.type == "cuda"
        else nullcontext()
    )

    from esm.modeling_esm import resolve_tokenizer_dir
    from esm.tokenizer import get_token_bytes, get_tokenizer

    tokenizer_path = resolve_tokenizer_dir(
        str(args.esm_ckpt), getattr(args, "tokenizer_path", None)
    )
    args.tokenizer_path = tokenizer_path
    token_bytes = get_token_bytes(device=device, tokenizer_dir=tokenizer_path)
    tokenizer = get_tokenizer(tokenizer_dir=tokenizer_path)

    if args.virtual_world_size > 1:
        total_batches = args.eval_batches * args.virtual_world_size
        print(
            f"Collecting {args.eval_batches} {args.split} batch(es) per virtual rank "
            f"from dataset={args.dataset} ({args.virtual_world_size} ranks, {total_batches} total)"
        )
    else:
        print(
            f"Collecting {args.eval_batches} {args.split} batch(es) from dataset={args.dataset}"
        )
    batches = collect_eval_batches(
        tokenizer,
        batch_size=args.device_batch_size,
        context_length=args.context_length,
        split=args.split,
        device=torch.device("cpu"),
        eval_batches=args.eval_batches,
        dataset=args.dataset,
        data_dir=args.data_dir,
        virtual_world_size=args.virtual_world_size,
    )

    results = {}
    model_meta = {}

    print(f"Loading ESM from {args.esm_ckpt}")
    esm_model, _, esm_hparams = load_esm(args, device)
    if args.context_length != int(
        esm_hparams.get("context_length", args.context_length)
    ):
        print(
            f"[warning] requested context_length={args.context_length}, "
            f"ESM checkpoint context_length={esm_hparams.get('context_length')}"
        )

    print("Evaluating ESM position-wise BPB")
    results["esm"] = compute_position_bpb(
        "esm",
        batches,
        loss2d_fn=lambda x, y, model=esm_model: esm_final_step_loss2d(model, x, y),
        device=device,
        context_length=args.context_length,
        token_bytes=token_bytes,
        autocast_ctx=autocast_ctx,
    )
    model_meta["esm"] = {
        "checkpoint": str(args.esm_ckpt.resolve()),
        "model_name": esm_hparams.get("model_name"),
        "model_size": esm_hparams.get("model_size"),
        "overall_bpb": results["esm"]["overall_bpb"],
        "mean_position_bpb": results["esm"]["mean_position_bpb"],
    }
    del esm_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    out = {
        "context_length": args.context_length,
        "split": args.split,
        "dataset": args.dataset,
        "data_dir": str(args.data_dir.resolve()) if args.data_dir is not None else None,
        "device_batch_size": args.device_batch_size,
        "eval_batches": args.eval_batches,
        "virtual_world_size": args.virtual_world_size,
        "total_eval_batches": len(batches),
        "num_samples": next(iter(results.values()))["num_samples"],
        "models": model_meta,
        "mean_position_bpb": {
            name: result["mean_position_bpb"] for name, result in results.items()
        },
        "position_bpb": {
            name: result["position_bpb"] for name, result in results.items()
        },
        "logit": {name: result["logit"] for name, result in results.items()},
        "position_nats": {
            name: result["position_nats"] for name, result in results.items()
        },
        "position_bytes": {
            name: result["position_bytes"] for name, result in results.items()
        },
        "position_token_count": {
            name: result["position_token_count"] for name, result in results.items()
        },
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"Wrote {args.output_json}")
    for name, result in results.items():
        print(f"{name} overall BPB: {result['overall_bpb']}")
        print(f"{name} mean position BPB: {result['mean_position_bpb']}")
    return out


def load_model(ckpt_path, tokenizer_path, device, dtype=None):
    """Load a checkpoint through the shared checkpoint loader."""
    _load_core_runtime()
    if dtype is None:
        dtype = torch.bfloat16
    from esm.modeling_esm import load_checkpoint

    model, tokenizer, hparams, _ = load_checkpoint(
        ckpt_path,
        device=device,
        dtype=dtype,
        tokenizer_path=tokenizer_path or None,
    )
    return model, tokenizer, hparams


def evaluate_task(model, tokenizer, data, device, task_meta, max_seq_len=None):
    """Evaluate a task through the shared ESM evaluator."""
    _load_core_runtime()
    return _esm_evaluate_task(model, tokenizer, data, device, task_meta)


@_no_grad
def evaluate_example_with_trajectory(idx, model, tokenizer, data, device, task_meta):
    """
    Evaluate one example and return the intermediate trajectory.

    The implementation reuses the shared rendering, batching, and forward
    helpers.

    Returns:
        (is_correct: bool, trajectory: dict)
    """
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(available_indices, num_fewshot)
        fewshot_examples = [data[i] for i in fewshot_indices]

    if task_type == "multiple_choice":
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)
    elif task_type == "schema":
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)
    elif task_type == "language_modeling":
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    if hasattr(model, "max_seq_len") and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []
        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = len(t) - max_tokens
                new_tokens.append(t[-max_tokens:])
                new_start_idxs.append(s - num_to_crop)
                new_end_idxs.append(e - num_to_crop)
                assert s - num_to_crop >= 0
                assert e - num_to_crop >= 0
            else:
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)
        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    pad_token_id = tokenizer.get_bos_token_id()
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)

    losses, predictions = forward_model(model, input_ids)

    trajectory = {
        "sample_id": idx,
        "is_correct": False,
        "metadata": {
            "task_type": task_type,
            "num_fewshot": num_fewshot,
        },
    }

    if task_type == "language_modeling":
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si - 1 : ei - 1]
        actual_tokens = input_ids[0, si:ei]
        is_correct = torch.all(predicted_tokens == actual_tokens).item()

        try:
            gt_text = tokenizer.decode(actual_tokens.tolist())
        except Exception:
            gt_text = str(actual_tokens.tolist())
        try:
            pred_text = tokenizer.decode(predicted_tokens.tolist())
        except Exception:
            pred_text = str(predicted_tokens.tolist())

        trajectory["input"] = prompts[0]
        trajectory["ground_truth"] = gt_text
        trajectory["prediction"] = pred_text
        trajectory["parsed_answer"] = pred_text

    elif task_type in ["multiple_choice", "schema"]:
        mean_losses = [
            losses[i, si - 1 : ei - 1].mean().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))
        gold_idx = item["gold"]
        is_correct = pred_idx == gold_idx

        if task_type == "multiple_choice":
            choices = item.get("choices", [])
            trajectory["input"] = (
                prompts[gold_idx] if gold_idx < len(prompts) else prompts[0]
            )
            trajectory["ground_truth"] = (
                choices[gold_idx] if gold_idx < len(choices) else str(gold_idx)
            )
            trajectory["prediction"] = (
                choices[pred_idx] if pred_idx < len(choices) else str(pred_idx)
            )
            trajectory["parsed_answer"] = pred_idx
            trajectory["metadata"]["num_choices"] = len(choices)
        else:
            context_options = item.get("context_options", [])
            continuation = item.get("continuation", "")
            trajectory["input"] = (
                prompts[gold_idx] if gold_idx < len(prompts) else prompts[0]
            )
            trajectory["ground_truth"] = (
                context_options[gold_idx]
                if gold_idx < len(context_options)
                else str(gold_idx)
            )
            trajectory["prediction"] = (
                context_options[pred_idx]
                if pred_idx < len(context_options)
                else str(pred_idx)
            )
            trajectory["parsed_answer"] = pred_idx
            trajectory["metadata"]["num_choices"] = len(context_options)
            trajectory["metadata"]["continuation"] = continuation

        trajectory["metadata"]["mean_losses"] = mean_losses
        trajectory["metadata"]["gold_idx"] = gold_idx
        trajectory["metadata"]["pred_idx"] = pred_idx
    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    trajectory["is_correct"] = bool(is_correct)
    return is_correct, trajectory


def evaluate_task_with_trajectory(
    model,
    tokenizer,
    data,
    device,
    task_meta,
    output_dir=None,
    task_label=None,
    num_trajectory_samples=10,
    show_progress=True,
    progress_position=1,
):
    """
    Evaluate a task while collecting trajectories for selected examples.

    The first ``num_trajectory_samples`` examples use the trajectory path;
    remaining examples use the standard evaluator.

    Returns:
        (mean_correct: float, trajectories: list[dict])
    """
    n = len(data)
    correct = 0

    trajectories = []
    traj_file = None
    if num_trajectory_samples > 0 and output_dir and task_label:
        traj_dir = os.path.join(output_dir, "trajectories", task_label)
        os.makedirs(traj_dir, exist_ok=True)
        traj_path = os.path.join(traj_dir, "samples.jsonl")
        traj_file = open(traj_path, "w", encoding="utf-8")

    iterator = range(n)
    if show_progress:
        iterator = tqdm(
            iterator,
            desc="  samples",
            position=progress_position,
            leave=False,
            unit="ex",
        )

    try:
        for idx in iterator:
            if num_trajectory_samples > 0 and idx < num_trajectory_samples:
                is_correct, traj = evaluate_example_with_trajectory(
                    idx, model, tokenizer, data, device, task_meta
                )
                trajectories.append(traj)
                if traj_file:
                    traj_file.write(json.dumps(traj, ensure_ascii=False) + "\n")
                    traj_file.flush()
            else:
                is_correct = evaluate_example(
                    idx, model, tokenizer, data, device, task_meta
                )
            correct += int(is_correct)
    finally:
        if traj_file:
            traj_file.close()

    mean_correct = correct / n if n > 0 else 0.0
    return mean_correct, trajectories


def evaluate_core(
    model,
    tokenizer,
    device,
    eval_bundle_dir,
    max_per_task=-1,
    task_samples=None,
    output_dir=None,
    num_trajectory_samples=10,
):
    """
    Run the complete CORE evaluation.

    Args:
        model: Model to evaluate.
        tokenizer: Tokenizer used by the model.
        device: Evaluation device.
        eval_bundle_dir: Directory containing the evaluation bundle.
        max_per_task: Default sample limit per task; ``-1`` means all samples.
        task_samples: Optional per-task sample limits.
        output_dir: Directory for trajectory files.
        num_trajectory_samples: Number of trajectories to save per task.
    """
    _load_core_runtime()
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    data_base_path = os.path.join(eval_bundle_dir, "eval_data")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    tasks = config["icl_tasks"]

    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            task_name = row["Eval Task"]
            random_baseline = row["Random baseline"]
            random_baselines[task_name] = float(random_baseline)

    results = {}
    centered_results = {}
    timing_records = []

    print("\n" + "=" * 80)
    print("Running CORE Evaluation")
    print("=" * 80 + "\n")

    task_pbar = tqdm(tasks, desc="Tasks", position=0, unit="task")
    for task_idx, task in enumerate(task_pbar):
        start_time = time.time()
        task_start_iso = datetime.now().isoformat()
        label = task["label"]
        task_pbar.set_description(f"[{task_idx + 1}/{len(tasks)}] {label}")

        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }

        tqdm.write(f"[{task_idx + 1}/{len(tasks)}] Evaluating: {label}")
        tqdm.write(
            f"  Type: {task_meta['task_type']} | Few-shot: {task_meta['num_fewshot']}"
        )

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]
            tqdm.write(f"  Using task-specific sample limit: {task_max_samples}")

        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]
            tqdm.write(f"  Using {len(data)} samples (subsampled)")
        else:
            tqdm.write(f"  Total samples: {len(data)} (evaluating ALL samples)")

        accuracy, _ = evaluate_task_with_trajectory(
            model,
            tokenizer,
            data,
            device,
            task_meta,
            output_dir=output_dir,
            task_label=label,
            num_trajectory_samples=num_trajectory_samples,
            show_progress=True,
            progress_position=1,
        )
        results[label] = accuracy

        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (
            1.0 - 0.01 * random_baseline
        )
        centered_results[label] = centered_result

        elapsed = time.time() - start_time
        task_end_iso = datetime.now().isoformat()
        timing_records.append(
            {
                "task": label,
                "start_time": task_start_iso,
                "end_time": task_end_iso,
                "duration_seconds": round(elapsed, 2),
                "num_samples": len(data),
            }
        )
        tqdm.write(
            f"  -> Accuracy: {accuracy:.4f} | Centered: {centered_result:.4f} | Time: {elapsed:.2f}s\n"
        )

    task_pbar.close()

    core_metric = sum(centered_results.values()) / len(centered_results)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "timing": timing_records,
    }


def _greedy_assign_tasks(all_tasks, num_gpus, history_timing):
    """
    Assign the heaviest tasks to the least-loaded GPU.

    Falls back to round-robin assignment when timing history is unavailable.

    Args:
        all_tasks: Task entries from ``core.yaml``.
        num_gpus: Number of GPUs.
        history_timing: Timing summary data, or ``None``.

    Returns:
        A mapping from GPU rank to assigned task indices.
    """

    per_task_duration = {}
    if history_timing is not None:
        per_task = history_timing.get("per_task", {})
        if not per_task:
            if isinstance(history_timing, list):
                per_task = {
                    rec["task"]: rec["duration_seconds"]
                    for rec in history_timing
                    if "task" in rec
                }
            elif "timing" in history_timing:
                per_task = {
                    rec["task"]: rec["duration_seconds"]
                    for rec in history_timing["timing"]
                    if "task" in rec
                }
            elif "records" in history_timing:
                per_task = {
                    rec["task"]: rec["duration_seconds"]
                    for rec in history_timing["records"]
                    if "task" in rec
                }

        for label, val in per_task.items():
            if isinstance(val, dict):
                per_task_duration[label] = val.get("duration_seconds", 0)
            else:
                per_task_duration[label] = float(val)

    if not per_task_duration:
        assignment = {r: [] for r in range(num_gpus)}
        for i in range(len(all_tasks)):
            assignment[i % num_gpus].append(i)
        return assignment, False

    known_durations = list(per_task_duration.values())
    median_duration = (
        sorted(known_durations)[len(known_durations) // 2] if known_durations else 60.0
    )

    task_costs = []
    for i, task in enumerate(all_tasks):
        label = task["label"]
        cost = per_task_duration.get(label, median_duration)
        task_costs.append((i, label, cost))

    task_costs.sort(key=lambda x: x[2], reverse=True)

    gpu_heap = [(0.0, r) for r in range(num_gpus)]
    heapq.heapify(gpu_heap)
    assignment = {r: [] for r in range(num_gpus)}

    for task_idx, label, cost in task_costs:
        min_cost, min_rank = heapq.heappop(gpu_heap)
        assignment[min_rank].append(task_idx)
        heapq.heappush(gpu_heap, (min_cost + cost, min_rank))

    return assignment, True


def _eval_worker(
    rank,
    world_size,
    ckpt_path,
    tokenizer_path,
    dtype,
    eval_bundle_dir,
    max_per_task,
    task_samples,
    all_tasks,
    random_baselines,
    result_dict,
    timing_list,
    output_dir,
    num_trajectory_samples,
    per_rank_indices=None,
):
    """Run the evaluation worker for one GPU.

    Args:
        per_rank_indices: Optional precomputed task assignment by rank.
    """
    _load_core_runtime()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.set_float32_matmul_precision("medium")

    if per_rank_indices is not None and rank in per_rank_indices:
        task_indices = list(per_rank_indices[rank])
    else:
        task_indices = [i for i in range(len(all_tasks)) if i % world_size == rank]

    print(f"[GPU {rank}] Loading model... ({len(task_indices)} tasks assigned)")
    sys.stdout.flush()
    model, tokenizer, hparams = load_model(
        ckpt_path, tokenizer_path, device, dtype=dtype
    )

    data_base_path = os.path.join(eval_bundle_dir, "eval_data")

    for task_idx in task_indices:
        task = all_tasks[task_idx]
        label = task["label"]
        start_time = time.time()
        task_start_iso = datetime.now().isoformat()

        task_meta = {
            "task_type": task["icl_task_type"],
            "dataset_uri": task["dataset_uri"],
            "num_fewshot": task["num_fewshot"][0],
            "continuation_delimiter": task.get("continuation_delimiter", " "),
        }

        print(f"[GPU {rank}] [{task_idx + 1}/{len(all_tasks)}] Evaluating: {label}")
        sys.stdout.flush()

        data_path = os.path.join(data_base_path, task_meta["dataset_uri"])
        with open(data_path, "r", encoding="utf-8") as f:
            data = [json.loads(line.strip()) for line in f]

        task_max_samples = max_per_task
        if task_samples and label in task_samples:
            task_max_samples = task_samples[label]

        if task_max_samples > 0:
            shuffle_rng = random.Random(1337)
            shuffle_rng.shuffle(data)
            data = data[:task_max_samples]

        traj_file = None
        if num_trajectory_samples > 0 and output_dir:
            traj_dir = os.path.join(output_dir, "trajectories", label)
            os.makedirs(traj_dir, exist_ok=True)
            traj_path = os.path.join(traj_dir, f"samples_rank{rank}.jsonl")
            traj_file = open(traj_path, "w", encoding="utf-8")

        n = len(data)
        correct = 0
        try:
            for idx in range(n):
                if num_trajectory_samples > 0 and idx < num_trajectory_samples:
                    is_correct, traj = evaluate_example_with_trajectory(
                        idx, model, tokenizer, data, device, task_meta
                    )
                    if traj_file:
                        traj_file.write(json.dumps(traj, ensure_ascii=False) + "\n")
                        traj_file.flush()
                else:
                    is_correct = evaluate_example(
                        idx, model, tokenizer, data, device, task_meta
                    )
                correct += int(is_correct)
        finally:
            if traj_file:
                traj_file.close()

        accuracy = correct / n if n > 0 else 0.0

        random_baseline = random_baselines.get(label, 50.0)
        centered_result = (accuracy - 0.01 * random_baseline) / (
            1.0 - 0.01 * random_baseline
        )

        elapsed = time.time() - start_time
        task_end_iso = datetime.now().isoformat()
        print(
            f"[GPU {rank}]   -> {label}: acc={accuracy:.4f} centered={centered_result:.4f} ({elapsed:.2f}s)"
        )
        sys.stdout.flush()

        result_dict[label] = (accuracy, centered_result)
        timing_list.append(
            {
                "task": label,
                "start_time": task_start_iso,
                "end_time": task_end_iso,
                "duration_seconds": round(elapsed, 2),
                "num_samples": n,
                "gpu_rank": rank,
            }
        )


def evaluate_core_multigpu(
    num_gpus,
    ckpt_path,
    tokenizer_path,
    dtype,
    eval_bundle_dir,
    max_per_task,
    task_samples,
    output_dir=None,
    num_trajectory_samples=10,
    history_timing_path=None,
):
    """Run CORE evaluation across GPUs using load-balanced task assignment."""
    _load_core_runtime()
    config_path = os.path.join(eval_bundle_dir, "core.yaml")
    eval_meta_data = os.path.join(eval_bundle_dir, "eval_meta_data.csv")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    all_tasks = config["icl_tasks"]

    random_baselines = {}
    with open(eval_meta_data, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            random_baselines[row["Eval Task"]] = float(row["Random baseline"])

    history_timing = None
    if history_timing_path and os.path.isfile(history_timing_path):
        try:
            with open(history_timing_path, "r", encoding="utf-8") as f:
                history_timing = json.load(f)
        except Exception as e:
            print(f"  Warning: could not load history timing for scheduling: {e}")

    assignment, used_greedy = _greedy_assign_tasks(all_tasks, num_gpus, history_timing)

    print(f"\n{'=' * 80}")
    scheduling_method = (
        "greedy load-balanced" if used_greedy else "round-robin (no history timing)"
    )
    print(
        f"Running CORE Evaluation (multi-GPU: {num_gpus} GPUs, {len(all_tasks)} tasks)"
    )
    print(f"Task scheduling: {scheduling_method}")
    print(f"{'=' * 80}")
    for rank in range(num_gpus):
        task_names = [all_tasks[i]["label"] for i in assignment[rank]]

        if used_greedy and history_timing is not None:
            per_task = history_timing.get("per_task", {})
            if not per_task and "records" in history_timing:
                per_task = {
                    rec["task"]: rec
                    for rec in history_timing["records"]
                    if "task" in rec
                }
            total_est = 0
            for idx in assignment[rank]:
                label = all_tasks[idx]["label"]
                val = per_task.get(label, {})
                if isinstance(val, dict):
                    total_est += val.get("duration_seconds", 0)
                else:
                    total_est += float(val)
            print(
                f"  GPU {rank}: {len(task_names)} tasks (est. {total_est:.0f}s = {total_est / 60:.1f}min) -- {', '.join(task_names)}"
            )
        else:
            print(f"  GPU {rank}: {len(task_names)} tasks -- {', '.join(task_names)}")
    print()

    manager = mp.Manager()
    result_dict = manager.dict()
    timing_list = manager.list()

    per_rank_indices = manager.dict()
    for rank in range(num_gpus):
        per_rank_indices[rank] = assignment[rank]

    mp.spawn(
        _eval_worker,
        args=(
            num_gpus,
            ckpt_path,
            tokenizer_path,
            dtype,
            eval_bundle_dir,
            max_per_task,
            task_samples,
            all_tasks,
            random_baselines,
            result_dict,
            timing_list,
            output_dir,
            num_trajectory_samples,
            per_rank_indices,
        ),
        nprocs=num_gpus,
        join=True,
    )

    if num_trajectory_samples > 0 and output_dir:
        traj_base = os.path.join(output_dir, "trajectories")
        if os.path.isdir(traj_base):
            for task_dir_name in os.listdir(traj_base):
                task_dir = os.path.join(traj_base, task_dir_name)
                if not os.path.isdir(task_dir):
                    continue
                rank_files = sorted(
                    glob_module.glob(os.path.join(task_dir, "samples_rank*.jsonl"))
                )
                if rank_files:
                    merged_path = os.path.join(task_dir, "samples.jsonl")
                    with open(merged_path, "w", encoding="utf-8") as out_f:
                        for rf in rank_files:
                            with open(rf, "r", encoding="utf-8") as in_f:
                                for line in in_f:
                                    out_f.write(line)

                    for rf in rank_files:
                        os.remove(rf)

    results = {}
    centered_results = {}
    for label, (acc, centered) in result_dict.items():
        results[label] = acc
        centered_results[label] = centered

    core_metric = (
        sum(centered_results.values()) / len(centered_results)
        if centered_results
        else 0.0
    )

    timing_records = list(timing_list)

    return {
        "results": results,
        "centered_results": centered_results,
        "core_metric": core_metric,
        "timing": timing_records,
    }


def _find_latest_timing(esm_dir):
    """Find the latest timing summary under ``logs/core_eval``."""
    search_dir = os.path.join(esm_dir, "logs", "core_eval")
    if not os.path.isdir(search_dir):
        return None
    candidates = sorted(
        glob_module.glob(
            os.path.join(search_dir, "**/timing_summary.json"), recursive=True
        )
    )
    if not candidates:
        return None

    return max(candidates, key=os.path.getmtime)


def _print_eta_table(history_timing_path, tasks):
    """Load historical timing and print an ETA estimation table."""
    try:
        with open(history_timing_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception as e:
        print(f"  Warning: could not load history timing: {e}")
        return

    per_task = history.get("per_task", {})
    if not per_task:
        if isinstance(history, list):
            per_task = {
                rec["task"]: rec["duration_seconds"] for rec in history if "task" in rec
            }
        elif "timing" in history:
            per_task = {
                rec["task"]: rec["duration_seconds"]
                for rec in history["timing"]
                if "task" in rec
            }

    if not per_task:
        print(f"  Warning: no per-task timing found in {history_timing_path}")
        return

    task_labels = [t["label"] for t in tasks]
    total_est = 0
    print(f"\n  ETA Estimation (based on {history_timing_path}):")
    print(f"  {'Task':<40s} {'Est. Time':>10s}")
    print(f"  {'-' * 40} {'-' * 10}")
    for label in task_labels:
        est = per_task.get(label, {})
        if isinstance(est, dict):
            dur = est.get("duration_seconds", 0)
        else:
            dur = float(est)
        total_est += dur
        print(f"  {label:<40s} {dur:>8.1f}s")
    print(f"  {'-' * 40} {'-' * 10}")
    print(f"  {'TOTAL':<40s} {total_est:>8.1f}s  ({total_est / 60:.1f}min)")
    print()


def evaluate_core_cli(argv=None):
    parser = argparse.ArgumentParser(
        description="ESM CORE Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:

1. Evaluate all samples for the full QA score:
   python -m scripts.qa --task core --checkpoint model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task -1

2. Run a quick test with 100 samples per task:
   python -m scripts.qa --task core --checkpoint model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100

3. Set different sample counts for selected tasks:
   python -m scripts.qa --task core --checkpoint model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --max-per-task 100 \\
       --task-samples "hellaswag_zeroshot:500,arc_easy:200"

4. Use multiple GPUs:
   python -m scripts.qa --task core --checkpoint model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --gpus 4

5. Save trajectories for the first 20 samples of each task:
   python -m scripts.qa --task core --checkpoint model.ckpt --tokenizer-path tokenizer/ \\
       --eval-bundle-dir eval_bundle/ --output-dir output/ --num-trajectory-samples 20
        """,
    )
    parser.add_argument(
        "--ckpt-path", type=str, required=True, help="ESM checkpoint path"
    )
    parser.add_argument(
        "--tokenizer-path", type=str, required=True, help="Tokenizer path"
    )
    parser.add_argument(
        "--eval-bundle-dir",
        type=str,
        required=True,
        help="Path to eval_bundle directory",
    )
    parser.add_argument(
        "--eval-modes",
        type=str,
        default="core",
        help="Evaluation modes (default: core)",
    )
    parser.add_argument(
        "--max-per-task",
        type=int,
        default=-1,
        help="Max examples per task globally (-1 = all samples for complete CORE score)",
    )
    parser.add_argument(
        "--task-samples",
        type=str,
        default="",
        help='Task-specific sample limits (format: "task1:num1,task2:num2")',
    )
    parser.add_argument(
        "--device-batch-size", type=int, default=16, help="Batch size per device"
    )
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Output directory"
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="Number of GPUs to use (-1 = auto-detect all available GPUs)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "bfloat16"],
        help="Model dtype for inference (default: bfloat16)",
    )
    parser.add_argument(
        "--num-trajectory-samples",
        type=int,
        default=10,
        help="Number of trajectory samples to save per task (0 = disable)",
    )
    parser.add_argument(
        "--history-timing",
        type=str,
        default="",
        help="Path to historical timing_summary.json for ETA estimation (auto-detect if empty)",
    )
    args = parser.parse_args(argv)
    _load_core_runtime()

    task_samples_dict = {}
    if args.task_samples:
        for item in args.task_samples.split(","):
            if ":" in item:
                task_name, num_samples = item.split(":", 1)
                task_samples_dict[task_name.strip()] = int(num_samples.strip())

    if args.gpus == -1:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"Auto-detected {num_gpus} GPU(s)")
        else:
            num_gpus = 0
            print("No GPU detected, using CPU")
    else:
        num_gpus = args.gpus
        print(f"Using {num_gpus} GPU(s) as specified")

    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    torch.set_float32_matmul_precision("medium")

    os.makedirs(args.output_dir, exist_ok=True)

    esm_dir = str(Path(__file__).parent.parent)
    history_timing_path = args.history_timing
    if not history_timing_path:
        history_timing_path = _find_latest_timing(esm_dir)
    if history_timing_path and os.path.isfile(history_timing_path):
        print(f"\nFound historical timing: {history_timing_path}")

        config_path = os.path.join(args.eval_bundle_dir, "core.yaml")
        if os.path.isfile(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                _config = yaml.safe_load(f)
            _print_eta_table(history_timing_path, _config["icl_tasks"])
    else:
        print(
            "\nNo historical timing found (first run or --history-timing not specified)"
        )

    eval_start_time = time.time()
    eval_start_iso = datetime.now().isoformat()

    if num_gpus > 1 and "core" in args.eval_modes:
        print(f"\nUsing multi-GPU evaluation with {num_gpus} GPUs")
        core_results = evaluate_core_multigpu(
            num_gpus=num_gpus,
            ckpt_path=args.ckpt_path,
            tokenizer_path=args.tokenizer_path,
            dtype=dtype,
            eval_bundle_dir=args.eval_bundle_dir,
            max_per_task=args.max_per_task,
            task_samples=task_samples_dict if task_samples_dict else None,
            output_dir=args.output_dir,
            num_trajectory_samples=args.num_trajectory_samples,
            history_timing_path=history_timing_path,
        )
    else:
        if num_gpus >= 1:
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")

        print(f"Using device: {device}\n")

        print("=" * 80)
        print("Loading ESM Model")
        print("=" * 80 + "\n")

        model, tokenizer, hparams = load_model(
            args.ckpt_path, args.tokenizer_path, device, dtype=dtype
        )

        core_results = None
        if "core" in args.eval_modes:
            core_results = evaluate_core(
                model,
                tokenizer,
                device,
                args.eval_bundle_dir,
                max_per_task=args.max_per_task,
                task_samples=task_samples_dict if task_samples_dict else None,
                output_dir=args.output_dir,
                num_trajectory_samples=args.num_trajectory_samples,
            )

    eval_end_time = time.time()
    eval_end_iso = datetime.now().isoformat()
    total_seconds = round(eval_end_time - eval_start_time, 2)

    if core_results:
        print("\n" + "=" * 80)
        print("CORE Evaluation Results")
        print("=" * 80 + "\n")

        print(f"CORE Metric: {core_results['core_metric']:.4f}\n")

        print("Task Results:")
        for task_name, accuracy in sorted(core_results["results"].items()):
            centered = core_results["centered_results"][task_name]
            print(f"  {task_name:40s}: {accuracy:.4f} (centered: {centered:.4f})")

        results_to_save = {
            "results": core_results["results"],
            "centered_results": core_results["centered_results"],
            "core_metric": core_results["core_metric"],
        }
        results_file = os.path.join(args.output_dir, "core_results.json")
        with open(results_file, "w") as f:
            json.dump(results_to_save, f, indent=2)
        print(f"\n-> Results saved to: {results_file}")

        csv_file = os.path.join(args.output_dir, "core_results.csv")
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Task", "Accuracy", "Centered", "CORE"])
            for task_name in sorted(core_results["results"].keys()):
                writer.writerow(
                    [
                        task_name,
                        f"{core_results['results'][task_name]:.4f}",
                        f"{core_results['centered_results'][task_name]:.4f}",
                        "",
                    ]
                )
            writer.writerow(["OVERALL", "", "", f"{core_results['core_metric']:.4f}"])
        print(f"-> CSV saved to: {csv_file}")

        timing_records = core_results.get("timing", [])
        per_task_timing = {}
        for rec in timing_records:
            per_task_timing[rec["task"]] = {
                "duration_seconds": rec["duration_seconds"],
                "num_samples": rec.get("num_samples", -1),
                "start_time": rec.get("start_time", ""),
                "end_time": rec.get("end_time", ""),
            }
        timing_summary = {
            "total_seconds": total_seconds,
            "total_start": eval_start_iso,
            "total_end": eval_end_iso,
            "num_gpus": num_gpus if num_gpus > 1 else 1,
            "per_task": per_task_timing,
            "records": timing_records,
        }
        timing_file = os.path.join(args.output_dir, "timing_summary.json")
        with open(timing_file, "w") as f:
            json.dump(timing_summary, f, indent=2)
        print(f"-> Timing saved to: {timing_file}")

        traj_dir = os.path.join(args.output_dir, "trajectories")
        if os.path.isdir(traj_dir):
            traj_tasks = [
                d
                for d in os.listdir(traj_dir)
                if os.path.isdir(os.path.join(traj_dir, d))
            ]
            print(f"-> Trajectories saved for {len(traj_tasks)} tasks in: {traj_dir}/")

        num_tasks = len(core_results["results"])
        avg_acc = np.mean(list(core_results["results"].values()))
        print(
            f"\n[EVAL_SUMMARY] dataset=core core_metric={core_results['core_metric']:.4f} avg_accuracy={avg_acc:.4f} num_tasks={num_tasks} total_time={total_seconds:.1f}s"
        )

    print("\n" + "=" * 80)
    print("Evaluation Complete!")
    print("=" * 80 + "\n")
    return core_results


def main(argv: list[str] | None = None) -> None:
    parser = _parser()
    args = parse_with_config(parser, kind="eval", argv=argv)
    args = _resolve_eval_paths(args)
    args.device_batch_size = args.batch_size_per_device
    args.grad_accum = args.accumulate_grad_batches
    print_resolved_config(args, kind="eval")
    return evaluate(args)


if __name__ == "__main__":
    main()
