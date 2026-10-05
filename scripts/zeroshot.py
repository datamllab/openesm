"""Run the MDLM Table-3 zero-shot evaluation for an ESM checkpoint."""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch

from esm.modeling_esm import (
    get_token_bytes,
    get_tokenizer,
    load_checkpoint,
    load_checkpoint_file,
)
from scripts.zeroshot_datasets import (
    DATA_ROOT_DEFAULT,
    EVAL_ORDER,
    build_blocks,
)


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def prepare_slim_checkpoint(source_path, target_path):
    if os.path.isfile(target_path):
        log(f"using cached slim checkpoint {target_path}")
        return
    log(f"loading checkpoint for slim copy: {source_path}")
    checkpoint = load_checkpoint_file(source_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith("model.transformer_eager.")
        and not key.startswith("transformer_eager.")
    }
    hparams = dict(checkpoint["hyper_parameters"])
    hparams.pop("tokenizer_obj", None)
    slim = {
        "hyper_parameters": hparams,
        "state_dict": state_dict,
    }
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    torch.save(slim, target_path)
    log(f"saved slim checkpoint {target_path}")


def evaluate_blocks(model, blocks, batch_size, device, token_bytes):
    nll_sum = 0.0
    token_sum = 0
    bpb_nats = 0.0
    bpb_bytes = 0
    start = time.time()
    model_dtype = model.embeddings.weight.dtype
    autocast_enabled = device.type == "cuda" and model_dtype in (
        torch.float16,
        torch.bfloat16,
    )
    for offset in range(0, len(blocks), batch_size):
        batch = torch.from_numpy(blocks[offset : offset + batch_size]).to(device)
        inputs = batch[:, :-1].unsqueeze(0)
        targets = batch[:, 1:].unsqueeze(0)
        with torch.enable_grad(), torch.amp.autocast(
            device_type=device.type,
            dtype=model_dtype,
            enabled=autocast_enabled,
        ):
            metrics = model.forward_loss_wrapper(
                (inputs, targets), phase="valid", token_bytes=token_bytes
            )
        supervised_tokens = int(metrics["supervised_tokens"])
        nll_sum += float(metrics["final_step_loss"]) * supervised_tokens
        token_sum += supervised_tokens
        bpb_nats += float(metrics["bpb_nats"])
        bpb_bytes += int(metrics["bpb_bytes"])
        if offset == 0 or offset + batch_size >= len(blocks):
            log(f"evaluated {min(offset + batch_size, len(blocks))}/{len(blocks)} blocks")
    mean_nll = nll_sum / max(token_sum, 1)
    return {
        "mean_nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 80.0)),
        "nll_sum": nll_sum,
        "supervised_tokens": token_sum,
        "bpb_nats": bpb_nats,
        "bpb_bytes": bpb_bytes,
        "bpb": bpb_nats / (math.log(2) * bpb_bytes) if bpb_bytes else float("nan"),
        "n_blocks": len(blocks),
        "seconds": time.time() - start,
    }


def merge_shards(dataset, output_dir):
    paths = sorted(glob.glob(os.path.join(output_dir, "shards", f"zeroshot_{dataset}_shard*.json")))
    if not paths:
        raise FileNotFoundError(f"no shard results found for {dataset}")
    records = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            records.append(json.load(handle))
    shard_count = records[0]["num_shards"]
    shard_ids = sorted(record["shard_index"] for record in records)
    if shard_ids != list(range(shard_count)):
        raise RuntimeError(f"{dataset}: incomplete shards {shard_ids}; expected {shard_count}")
    nll_sum = sum(record["nll_sum"] for record in records)
    token_sum = sum(record["supervised_tokens"] for record in records)
    bpb_nats = sum(record["bpb_nats"] for record in records)
    bpb_bytes = sum(record["bpb_bytes"] for record in records)
    mean_nll = nll_sum / max(token_sum, 1)
    result = {
        "model": records[0]["model"],
        "dataset": dataset,
        "checkpoint": records[0]["checkpoint"],
        "source_checkpoint": records[0]["source_checkpoint"],
        "block_size": records[0]["block_size"],
        "batch_size": records[0]["batch_size"],
        "protocol": records[0]["protocol"],
        "mean_nll": mean_nll,
        "ppl": math.exp(min(mean_nll, 80.0)),
        "nll_sum": nll_sum,
        "supervised_tokens": token_sum,
        "bpb_nats": bpb_nats,
        "bpb_bytes": bpb_bytes,
        "bpb": bpb_nats / (math.log(2) * bpb_bytes) if bpb_bytes else float("nan"),
        "n_blocks": sum(record["n_blocks"] for record in records),
        "num_shards": shard_count,
    }
    path = os.path.join(output_dir, f"zeroshot_{dataset}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    log(f"merged {dataset}: PPL={result['ppl']:.4f} BPB={result['bpb']:.4f}")


def aggregate(output_dir):
    results = {}
    for path in glob.glob(os.path.join(output_dir, "zeroshot_*.json")):
        with open(path, encoding="utf-8") as handle:
            record = json.load(handle)
        results[record["dataset"]] = record
    ordered = {name: results[name] for name in EVAL_ORDER if name in results}
    if len(ordered) != len(EVAL_ORDER):
        raise RuntimeError(f"missing zero-shot results: {set(EVAL_ORDER) - set(ordered)}")
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, indent=2)
    lines = [
        "# ESM zero-shot evaluation",
        "",
        "MDLM Table-3 protocol with the ESM tokenizer and token-weighted metrics.",
        "",
        "| dataset | PPL | BPB | mean NLL | blocks | tokens |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, record in ordered.items():
        lines.append(
            f"| {name} | {record['ppl']:.4f} | {record['bpb']:.4f} | "
            f"{record['mean_nll']:.4f} | {record['n_blocks']} | "
            f"{record['supervised_tokens']} |"
        )
    with open(os.path.join(output_dir, "summary.md"), "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def evaluate_dataset(args):
    tokenizer_dir = os.path.abspath(args.tokenizer_dir)
    wrapped, tokenizer, hparams, device = load_checkpoint(
        args.checkpoint,
        device=args.device,
        dtype=args.dtype,
        tokenizer_path=tokenizer_dir,
    )
    token_bytes = get_token_bytes(device=device, tokenizer_dir=tokenizer_dir)
    model = wrapped.model
    model.eval()
    torch.manual_seed(0)
    blocks = build_blocks(
        args.dataset,
        tokenizer,
        args.data_root,
        args.block_size,
        args.cache_dir,
        log,
    )
    if args.shard_index is not None:
        blocks = blocks[args.shard_index :: args.num_shards]
    metrics = evaluate_blocks(model, blocks, args.batch_size, device, token_bytes)
    result = {
        "model": hparams.get("model_name", "esm"),
        "dataset": args.dataset,
        "checkpoint": args.checkpoint,
        "source_checkpoint": args.source_checkpoint or args.checkpoint,
        "block_size": args.block_size,
        "batch_size": args.batch_size,
        "protocol": "MDLM Table-3: local data, detokenization, BOS/EOS wrapping, AR shift",
        **metrics,
    }
    if args.shard_index is not None:
        result["shard_index"] = args.shard_index
        result["num_shards"] = args.num_shards
        output_path = os.path.join(
            args.output_dir,
            "shards",
            f"zeroshot_{args.dataset}_shard{args.shard_index}.json",
        )
    else:
        output_path = os.path.join(args.output_dir, f"zeroshot_{args.dataset}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    log(f"{args.dataset}: PPL={metrics['ppl']:.4f} BPB={metrics['bpb']:.4f} -> {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=False)
    parser.add_argument("--source-checkpoint", default="")
    parser.add_argument("--slim-checkpoint", default="")
    parser.add_argument("--prepare-checkpoint", action="store_true")
    parser.add_argument("--dataset", choices=EVAL_ORDER)
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--tokenizer-dir", required=False)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--block-size", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--merge-shards", action="store_true")
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        return args
    if args.merge_shards:
        if not args.dataset:
            parser.error("--dataset is required with --merge-shards")
        return args
    if args.prepare_checkpoint:
        if not args.checkpoint or not args.slim_checkpoint:
            parser.error("--checkpoint and --slim-checkpoint are required")
        return args
    if args.prepare_only:
        if not args.tokenizer_dir or not args.dataset:
            parser.error("--tokenizer-dir and --dataset are required with --prepare-only")
        return args
    if not args.checkpoint or not args.tokenizer_dir or not args.dataset:
        parser.error("--checkpoint, --tokenizer-dir, and --dataset are required")
    return args


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    if args.aggregate:
        aggregate(args.output_dir)
    elif args.merge_shards:
        merge_shards(args.dataset, args.output_dir)
    elif args.prepare_checkpoint:
        prepare_slim_checkpoint(args.checkpoint, args.slim_checkpoint)
    elif args.prepare_only:
        tokenizer = get_tokenizer(args.tokenizer_dir)
        build_blocks(
            args.dataset,
            tokenizer,
            args.data_root,
            args.block_size,
            args.cache_dir,
            log,
        )
    else:
        evaluate_dataset(args)


if __name__ == "__main__":
    main()
