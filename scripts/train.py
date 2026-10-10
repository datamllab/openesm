import json
import os
import random
import shutil
import sys
from argparse import ArgumentParser

import torch


try:
    from esm.config import bootstrap_assets

    bootstrap_assets("train")
except Exception:
    pass


try:
    from esm.tokenizer import RustBPETokenizer

    torch.serialization.add_safe_globals([RustBPETokenizer])
except Exception:
    pass


if hasattr(torch.autograd.graph, "set_warn_on_accumulate_grad_stream_mismatch"):
    torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)

try:
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import ModelSummary
    from lightning.pytorch.loggers import WandbLogger
    from lightning.pytorch.strategies import DDPStrategy

    try:
        from lightning.pytorch.strategies import DeepSpeedStrategy
    except ImportError:
        DeepSpeedStrategy = None
    from lightning.pytorch.utilities.rank_zero import rank_zero_only
except ImportError:
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelSummary
    from pytorch_lightning.loggers import WandbLogger
    from pytorch_lightning.strategies import DDPStrategy

    try:
        from pytorch_lightning.strategies import DeepSpeedStrategy
    except ImportError:
        DeepSpeedStrategy = None
    from pytorch_lightning.utilities.rank_zero import rank_zero_only

from esm import logger as text_logger
from esm.disk_aware_checkpoint import DiskAwareCheckpoint, DiskAwareFinalCheckpoint
from esm.config import parse_with_config, print_resolved_config
from esm.modeling_esm import init_wandb_watch, load_checkpoint, model_sizes


@rank_zero_only
def setup_wandb(args):
    import wandb

    wandb_dir = getattr(args, "wandb_save_dir", "logs/")
    os.makedirs(wandb_dir, exist_ok=True)
    if wandb.run is None:
        run = wandb.init(
            dir=wandb_dir,
            name=f"{args.run_name}",
            entity=f"{args.wandb_entity}",
            project=f"{args.wandb_project}",
            mode="offline" if args.wandb_offline else "online",
        )
        wandb.define_metric("__init", hidden=True)
        return run
    return None


def _write_resolved_config(args):
    """Persist the exact resolved arguments next to the run logs."""

    if int(os.environ.get("RANK", "0")) != 0:
        return
    path = os.path.join(args.run_dir, "resolved_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)


def _bytes_to_gib(num_bytes):
    return num_bytes / (1024**3)


def _checkpoint_file_size_gib(path):
    if path and os.path.isfile(path):
        return _bytes_to_gib(os.path.getsize(path))
    return None


def _largest_existing_checkpoint_gib(ckpt_dir):
    if not os.path.isdir(ckpt_dir):
        return None, None

    largest_size = None
    largest_path = None
    for root, _, files in os.walk(ckpt_dir):
        for filename in files:
            if not filename.endswith(".ckpt"):
                continue
            path = os.path.join(root, filename)
            try:
                size_gib = _bytes_to_gib(os.path.getsize(path))
            except OSError:
                continue
            if largest_size is None or size_gib > largest_size:
                largest_size = size_gib
                largest_path = path
    return largest_size, largest_path


def _estimate_checkpoint_size_gib(args, ckpt_dir, model_trainer):
    candidates = []

    existing_size_gib, existing_path = _largest_existing_checkpoint_gib(ckpt_dir)
    if existing_size_gib is not None:
        candidates.append((existing_size_gib, f"existing checkpoint: {existing_path}"))

    for attr_name, label in (
        ("finetuning_model_ckpt", "finetuning checkpoint"),
        ("resume_training_ckpt", "resume checkpoint"),
    ):
        path = getattr(args, attr_name, "")
        size_gib = _checkpoint_file_size_gib(path)
        if size_gib is not None:
            candidates.append((size_gib, f"{label}: {path}"))

    if candidates:
        return max(candidates, key=lambda item: item[0])

    tensor_bytes = 0
    for value in model_trainer.state_dict().values():
        if torch.is_tensor(value):
            tensor_bytes += value.numel() * value.element_size()

    estimated_gib = max(_bytes_to_gib(tensor_bytes) * 4.0, 0.1)
    return estimated_gib, "estimated from model state_dict"


def _final_checkpoint_enabled(args):

    return True


def _effective_top_k(args, final_checkpoint_enabled):
    if final_checkpoint_enabled and args.save_top_k_ckpts > 0:
        return max(args.save_top_k_ckpts - 1, 0)
    return args.save_top_k_ckpts


def _planned_checkpoint_slots(
    args, save_last, final_checkpoint_enabled, effective_top_k
):
    slots = 0
    details = []
    warnings = []

    if effective_top_k == -1:
        slots += 1
        details.append("top_k=all (preflight lower bound: 1)")
        warnings.append(
            "save_top_k_ckpts=-1 is unbounded, so exact checkpoint capacity cannot be guaranteed."
        )
    elif effective_top_k > 0:
        slots += effective_top_k
        details.append(f"top_k={effective_top_k}")

    if final_checkpoint_enabled:
        slots += 1
        details.append("final=1")

    if save_last:
        slots += 1
        details.append("last=1")

    if args.save_periodic_steps > 0:
        slots += 1
        details.append(f"periodic=1 every {args.save_periodic_steps} steps")

    return slots, details, warnings


def preflight_checkpoint_disk(
    args,
    ckpt_dir,
    save_last,
    model_trainer,
    is_rank_zero_process,
    final_checkpoint_enabled,
    effective_top_k,
):
    checkpoint_slots, slot_details, warnings = _planned_checkpoint_slots(
        args,
        save_last,
        final_checkpoint_enabled,
        effective_top_k,
    )
    if checkpoint_slots <= 0:
        return

    min_free_gib = max(0.0, float(args.checkpoint_min_free_gb))
    os.makedirs(ckpt_dir, exist_ok=True)

    free_gib = _bytes_to_gib(shutil.disk_usage(ckpt_dir).free)
    ref_ckpt_gib, ref_source = _estimate_checkpoint_size_gib(
        args, ckpt_dir, model_trainer
    )
    required_gib = checkpoint_slots * ref_ckpt_gib + min_free_gib

    if is_rank_zero_process:
        print("[Checkpoint Preflight]")
        print(f"  dir: {ckpt_dir}")
        print(
            f"  planned checkpoint slots: {checkpoint_slots} ({', '.join(slot_details)})"
        )
        print(f"  reference checkpoint size: {ref_ckpt_gib:.2f} GiB ({ref_source})")
        print(
            f"  required free before training: {required_gib:.2f} GiB = {checkpoint_slots} * {ref_ckpt_gib:.2f} + reserve {min_free_gib:.2f}"
        )
        print(f"  current free: {free_gib:.2f} GiB")
        for warning in warnings:
            print(f"  warning: {warning}")

    if free_gib < required_gib:
        raise RuntimeError(
            "Checkpoint disk preflight failed: "
            f"current free space {free_gib:.2f} GiB is less than required {required_gib:.2f} GiB "
            f"for {checkpoint_slots} checkpoint slots plus {min_free_gib:.2f} GiB reserve. "
            f"checkpoint_dir={ckpt_dir}; reference={ref_source}. "
            "Free disk space or reduce --save_top_k_ckpts/--save_periodic_steps before starting training."
        )


def main(args):

    from esm.trainer import ModelTrainer

    print(f"[output] output_root={args.output_root}")
    print(f"[output] run_name={args.run_name}")
    print(f"[output] run_dir={args.run_dir}")
    print(f"[output] log_root={args.log_root}")
    print(f"[output] checkpoint_dir={args.checkpoint_dir}")
    print(f"[output] wandb_save_dir={args.wandb_save_dir}")
    _write_resolved_config(args)

    if args.disable_wandb:
        args.no_wandb = True

    if args.is_random_seed:
        seed_everything(random.randint(0, 1000000), workers=True)
    else:
        seed_everything(33, workers=True)

    if args.debug_mode:
        args.no_wandb = True
        args.detect_anomaly = True
        args.limit_train_batches = 1

    if args.no_wandb:
        os.makedirs(args.log_root, exist_ok=True)

    wandb_logger = None
    if not args.no_wandb:
        run = None
        run = setup_wandb(args)

        wandb_logger = WandbLogger(
            save_dir=args.wandb_save_dir,
            name=f"{args.run_name}",
            entity=f"{args.wandb_entity}",
            project=f"{args.wandb_project}",
            offline=args.wandb_offline,
            experiment=run,
        )
        if args.wandb_tags is not None:
            wandb_logger.experiment.tags = args.wandb_tags
    else:
        console_log_file_path = os.path.join(args.log_root, args.console_log_filename)
        sys.stdout = text_logger.Tee(sys.stdout, console_log_file_path)
        sys.stderr = text_logger.Tee(sys.stderr, console_log_file_path)
        print(
            "$$$$$$$$$ NOTE THAT NOT ALL STDOUT LOGS (i.e. pytorch lighting logs) ARE CAPTURED THROUGH CONSOLE LOGGER, THIS IS ONLY RECOMMENDED FOR DEBUGGING $$$$$$$$$$"
        )

    if args.is_slurm_run:
        if not args.override_slurm_checks:
            assert (
                not args.debug_mode
                and not args.detect_anomaly
                and args.limit_train_batches == 1
                and args.limit_val_batches == 1
                and not args.find_unused_parameters
                and not args.debug_unused_parameters
            ), "for slurm run cannot have debug-only parameters enabled"

        print("Current Slurm job ID:", os.environ.get("SLURM_JOBID"))
        print("Current Slurm node list:", os.environ.get("SLURM_NODELIST"))
        print("SLURM_NTASKS:", os.environ.get("SLURM_NTASKS"))
        print("SLURM_GPUS_PER_NODE:", os.environ.get("SLURM_GPUS_PER_NODE"))
        print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

    try:
        model_config = model_sizes[args.model_size]
    except KeyError as exc:
        valid_sizes = ", ".join(model_sizes)
        raise ValueError(
            f"unsupported ESM model_size={args.model_size!r}; use one of: {valid_sizes}"
        ) from exc
    args.num_transformer_blocks = model_config["num_transformer_blocks"]
    args.multiheaded_attention_heads = model_config["multiheaded_attention_heads"]
    args.embedding_dim = model_config["embedding_dim"]
    print(
        "model_size",
        args.model_size,
        "args.num_transformer_blocks",
        args.num_transformer_blocks,
        "args.multiheaded_attention_heads",
        args.multiheaded_attention_heads,
        "args.embedding_dim",
        args.embedding_dim,
    )

    if args.modality == "NLP":
        assert args.embedding_dim != 0, "must define embedding dim for NLP models"
    else:
        raise ValueError(
            f"unsupported modality={args.modality!r}; ESM only supports the NLP/time-embedding path"
        )

    if not getattr(args, "time_embedding", True):
        raise ValueError(
            "ESM requires time_embedding=true; the non-time-embedding path was removed"
        )

    # num_nodes / proc_per_node / world_size are resolved once in esm/config.py
    print(
        f"num_nodes={args.num_nodes} proc_per_node={args.proc_per_node} "
        f"world_size={args.world_size}"
    )
    num_gpus = args.world_size
    # hand Lightning the resolved per-node device count
    args.gpus = args.proc_per_node
    print("devices/args.gpus: ", args.gpus)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    assert device == torch.device("cuda") and num_gpus > 0, (
        "using cpu instead of cuda. if you would like to proceed please remove this line and change code below to not use GPUs, otherwise check packages to ensure torch/others have cuda support"
    )
    print(f"GPU Availability: {device}, gpus: {num_gpus}\n")
    args.num_gpus = num_gpus
    effective_batch_size = (
        args.num_gpus * args.batch_size_per_device * args.accumulate_grad_batches
    )
    print(
        f"effective_batch_size: {effective_batch_size}",
        "batch_size_per_device",
        args.batch_size_per_device,
    )
    if args.lr_scaling_rule:
        scaled_lr = args.peak_learning_rate * effective_batch_size / 256
        args.peak_learning_rate = scaled_lr
        print(f"Learning Rate rescaled to: {scaled_lr} based off lr_scaling_rule")
    model_trainer = ModelTrainer(args)

    if args.finetuning_model_ckpt is not None and args.finetuning_model_ckpt != "":
        print(f"[SFT] Loading pretrained weights: {args.finetuning_model_ckpt}")
        pretrained, _, _, _ = load_checkpoint(
            args.finetuning_model_ckpt,
            device="cpu",
            dtype="float32",
            tokenizer_path=getattr(args, "tokenizer_dir", None),
        )
        pretrained_state = {
            f"model.{key}": value
            for key, value in pretrained.model.state_dict().items()
        }
        trainer_state = model_trainer.state_dict()
        compatible_state = {}
        matched_keys = set()
        for target_key in trainer_state:
            source_key = target_key.replace("._orig_mod.", ".")
            if source_key.startswith("model.transformer_eager."):
                source_key = source_key.replace(
                    "model.transformer_eager.", "model.transformer.", 1
                )
            if source_key in pretrained_state:
                compatible_state[target_key] = pretrained_state[source_key]
                matched_keys.add(source_key)

        unexpected_keys = sorted(set(pretrained_state) - matched_keys)
        if unexpected_keys:
            raise RuntimeError(
                "Unexpected pretrained parameter keys: "
                + ", ".join(unexpected_keys[:10])
            )
        incompatible = model_trainer.load_state_dict(compatible_state, strict=False)
        if incompatible.missing_keys:
            print(
                "[SFT] Parameters not initialized from the pretrained model: "
                + ", ".join(incompatible.missing_keys[:10])
            )
        print("[SFT] Weights loaded; training starts from step 0")

    is_rank_zero_process = int(os.environ.get("RANK", "0")) == 0
    if args.log_model_archi and is_rank_zero_process:
        print(str(model_trainer.model))
        print(str(args))

    if not args.no_wandb and args.wandb_watch:
        init_wandb_watch(
            wandb_logger,
            model_trainer,
            args.wandb_watch_log_freq,
            args.wandb_watch_level,
        )

    print(f"pytorch version: {torch.__version__}\n")

    if args.set_matmul_precision is not None:
        torch.set_float32_matmul_precision(args.set_matmul_precision)

    opt_name = args.optimizer if hasattr(args, "optimizer") else "adamw"
    token_budget_labels = {
        1_000_000_000: "1B",
        5_000_000_000: "5B",
        10_000_000_000: "10B",
    }
    target_total_tokens = getattr(args, "target_total_tokens", 0)
    token_budget_label = token_budget_labels.get(
        target_total_tokens,
        str(target_total_tokens) if target_total_tokens > 0 else "",
    )
    token_budget_suffix = f"-tokens{token_budget_label}" if token_budget_label else ""
    ckpt_dir = args.checkpoint_dir
    if not ckpt_dir:
        raise ValueError(
            "checkpoint_dir was not resolved; use output_root/run_name or --checkpoint_dir"
        )
    checkpoint_filename = f"s={{step}}-{args.model_size}{token_budget_suffix}-ctx{args.context_length}-lr{args.peak_learning_rate}-bs{args.batch_size_per_device}x{args.accumulate_grad_batches}-{opt_name}-{args.checkpoint_monitor_string}={{{args.checkpoint_monitor_string}:.4f}}"
    final_checkpoint_enabled = _final_checkpoint_enabled(args)
    effective_top_k = _effective_top_k(args, final_checkpoint_enabled)
    save_last = (
        not final_checkpoint_enabled
        and args.save_periodic_steps <= 0
        and args.save_top_k_ckpts != 0
    )
    checkpoint_cleanup_on_low_space = not args.disable_checkpoint_cleanup_on_low_space
    if args.checkpoint_preflight_required:
        preflight_checkpoint_disk(
            args,
            ckpt_dir,
            save_last,
            model_trainer,
            is_rank_zero_process,
            final_checkpoint_enabled,
            effective_top_k,
        )
    checkpoint_callback = DiskAwareCheckpoint(
        monitor=args.checkpoint_monitor_string,
        mode=args.checkpoint_monitor_mode,
        save_top_k=effective_top_k,
        save_last=save_last,
        dirpath=ckpt_dir,
        filename=checkpoint_filename,
        verbose=True,
        min_free_gb=args.checkpoint_min_free_gb,
        cleanup_on_low_space=checkpoint_cleanup_on_low_space,
    )

    periodic_checkpoint = None
    if args.save_periodic_steps > 0:
        periodic_checkpoint = DiskAwareCheckpoint(
            save_top_k=1,
            save_last=False,
            every_n_train_steps=args.save_periodic_steps,
            dirpath=ckpt_dir,
            filename=f"periodic-s={{step}}-{args.model_size}{token_budget_suffix}-ctx{args.context_length}",
            verbose=True,
            min_free_gb=args.checkpoint_min_free_gb,
            cleanup_on_low_space=checkpoint_cleanup_on_low_space,
        )

    final_checkpoint = None
    if final_checkpoint_enabled:
        final_checkpoint = DiskAwareFinalCheckpoint(
            dirpath=ckpt_dir,
            model_size=args.model_size,
            context_length=args.context_length,
            token_budget_label=token_budget_label,
            save_top_k=args.save_top_k_ckpts,
            monitor=args.checkpoint_monitor_string,
            mode=args.checkpoint_monitor_mode,
            min_free_gb=args.checkpoint_min_free_gb,
            cleanup_on_low_space=checkpoint_cleanup_on_low_space,
        )

    for name, param in model_trainer.model.named_parameters():
        if not param.requires_grad:
            print(f"Non-trainable parameters: {name} with shape {param.shape}")

    print("$$$$$$$$$$  STARTED TRAINING  $$$$$$$$$$")
    trainer = set_trainer(
        args,
        wandb_logger,
        checkpoint_callback,
        periodic_checkpoint=periodic_checkpoint,
        final_checkpoint=final_checkpoint,
    )
    resume_training_ckpt = (
        None if args.resume_training_ckpt == "" else args.resume_training_ckpt
    )
    trainer.fit(model_trainer, ckpt_path=resume_training_ckpt, weights_only=False)
    clear_cache()


def set_trainer(
    args,
    wandb_logger,
    checkpoint_callback,
    periodic_checkpoint=None,
    final_checkpoint=None,
):
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    if getattr(args, "use_deepspeed_stage3", False):
        if DeepSpeedStrategy is None:
            raise ImportError(
                "--use_deepspeed_stage3 requires the Lightning DeepSpeed strategy "
                "and the deepspeed package to be installed."
            )
        if args.find_unused_parameters:
            raise ValueError(
                "--use_deepspeed_stage3 is incompatible with --find_unused_parameters."
            )
        args.distributed_strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            offload_optimizer_device="cpu",
            offload_params_device="cpu",
            pin_memory=True,
            zero_allow_untested_optimizer=True,
        )

        if hasattr(args.distributed_strategy, "config"):
            args.distributed_strategy.config["zero_force_ds_cpu_optimizer"] = False
        print(
            "[distributed] DeepSpeed ZeRO Stage 3 enabled with CPU parameter/optimizer offload"
        )
    elif args.find_unused_parameters or args.distributed_strategy == "ddp":
        args.distributed_strategy = DDPStrategy(
            find_unused_parameters=True,
            gradient_as_bucket_view=True,
        )
    args.overfit_batches = (
        int(args.overfit_batches)
        if int(args.overfit_batches) == args.overfit_batches
        else args.overfit_batches
    )
    profiler = None if args.profiler == "" else args.profiler
    gradient_clip_val = args.gradient_clip_val if args.gradient_clip_val > 0 else None
    limit_val_batches = 0 if args.overfit_batches > 0 else args.limit_val_batches
    if (
        getattr(args, "dataset_name", "") == "esm_sft"
        and isinstance(args.val_check_interval, float)
        and args.val_check_interval > 1
        and args.val_check_interval.is_integer()
    ):
        args.val_check_interval = int(args.val_check_interval)
    callbacks = [checkpoint_callback]
    if periodic_checkpoint:
        callbacks.append(periodic_checkpoint)
    if final_checkpoint:
        callbacks.append(final_checkpoint)
    if args.log_model_archi:
        callbacks.append(ModelSummary(max_depth=2))

    trainer = Trainer(
        accelerator="auto",
        devices=args.gpus,
        num_nodes=args.num_nodes,
        precision=args.float_precision,
        max_steps=args.max_steps,
        logger=wandb_logger,
        enable_model_summary=args.log_model_archi,
        callbacks=callbacks,
        strategy=args.distributed_strategy,
        enable_checkpointing=True,
        fast_dev_run=args.fast_dev_run,
        num_sanity_val_steps=args.val_sanity,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=limit_val_batches,
        detect_anomaly=args.detect_anomaly,
        gradient_clip_val=gradient_clip_val,
        overfit_batches=args.overfit_batches,
        profiler=profiler,
        val_check_interval=args.val_check_interval,
        deterministic=args.deterministic,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        inference_mode=False,
    )

    return trainer


def clear_cache():
    torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument(
        "--output_root",
        type=str,
        default=".",
        help="Root for outputs; relative paths are resolved from the ESM project root.",
    )
    parser.add_argument(
        "--run_name",
        help="Single directory name under output_root/outputs (default: run_prefix-YYYYMMDD).",
        default="",
    )
    parser.add_argument(
        "--run_prefix",
        help="Prefix used when run_name is omitted.",
        default="pretrain-dclm-d6",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="",
        help="Override the run log directory; default: output_root/outputs/run_name/logs.",
    )

    parser.add_argument(
        "--modality",
        help="ESM training modality; only NLP is supported",
        choices=["NLP"],
        type=str,
        default="NLP",
    )

    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        lowered = str(value).lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
        raise ValueError(f"expected a boolean, got {value!r}")

    parser.add_argument(
        "--time_embedding",
        type=_parse_bool,
        default=True,
        help="ESM always uses the paper's time-embedding backbone",
    )

    parser.add_argument(
        "--model_name",
        choices=["esm"],
        default="esm",
        help="ESM model implementation (the only supported model is ESM)",
    )

    parser.add_argument(
        "--model_size",
        help="ESM model size, named by transformer depth (d{number})",
        choices=model_sizes.keys(),
        default="d6",
    )

    parser.add_argument(
        "--tokenizer", help="ESM tokenizer identifier", type=str, default="esm_bpe"
    )

    parser.add_argument(
        "--pretokenize_dataset",
        help="whether to pretokenize the dataset and save that or just tokenize as loading the dataset. tokenizing the dataset takes a long time and may need to be done using debug dataloader. due to a bug with HF does not always work reliably, may get stuck. not currently implemented for fine-tuning datasets",
        action="store_true",
        default=False,
    )

    parser.add_argument("--mcmc_step_size", type=float, default=300.0)
    parser.add_argument(
        "--mcmc_step_size_learnable", action="store_true", default=False
    )
    parser.add_argument("--mcmc_step_size_lr_multiplier", type=float, default=1500.0)
    parser.add_argument("--mcmc_num_steps", type=int, default=1)
    parser.add_argument(
        "--denoising_initial_condition", type=str, default="random_noise"
    )
    parser.add_argument("--gaussian_random_noise_scaling", type=float, default=1.0)
    parser.add_argument("--free_embed_noise_scale", type=float, default=1.0)
    parser.add_argument("--use_tf_head", action="store_true", default=False)
    parser.add_argument(
        "--tf_head_type", choices=["direct_unembed"], type=str, default="direct_unembed"
    )
    parser.add_argument("--free_embedding_mcmc", action="store_true", default=False)
    parser.add_argument("--esm_norm", choices=["rms"], type=str, default="rms")
    parser.add_argument("--esm_act_func", type=str, default="silu")
    parser.add_argument("--truncate_mcmc", action="store_true", default=False)

    parser.add_argument(
        "--context_length",
        help="Number of tokens in each training sequence",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--ffn_dim_multiplier",
        help="how much wider than the embedding dim the transformer FFN dim should be",
        type=float,
        default=None,
    )

    parser.add_argument(
        "--weight_initialization_method",
        help="xavier or he",
        type=str,
        default="xavier",
    )

    parser.add_argument(
        "--weight_initialization_gain",
        help="gain of the weight init, see https://pytorch.org/docs/stable/nn.init.html, default is equiavalent to linear gain which is 1. tried tweaking this for ESM didnt help",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gpus",
        type=int,
        default=-1,
        help="GPUs per node (-1 = every visible device)",
    )

    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="",
        help="value for CUDA_VISIBLE_DEVICES; a value already in the environment wins",
    )

    parser.add_argument(
        "--node_count",
        type=int,
        default=1,
        help="number of nodes; NODE_COUNT / SLURM / torchrun env take precedence",
    )

    parser.add_argument(
        "--distributed_strategy",
        help="distributed strategy - ddp_spawn, ddp, fsdp_native, or None",
        default="ddp",
    )

    parser.add_argument(
        "--peak_learning_rate",
        help="peak learning rate after warm up",
        type=float,
        default=0.02,
    )

    parser.add_argument(
        "--batch_size_per_device",
        help="batch size (PER DEVICE!!!, not effective). to get effective_batch_size is num_gpus * batch_size_per_device * accumulate_grad_batches, see above)",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--global_batch_size",
        type=int,
        default=0,
        help="resolved global sequence batch; checked against distributed world size",
    )

    parser.add_argument(
        "--gradient_clip_val",
        help="maximum value for gradient clipping",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--is_random_seed", help="is_random_seed", action="store_true", default=False
    )

    parser.add_argument(
        "--deterministic",
        help="ensures results are determnistic, may run a bit slower, sets flag on pl trainer and sets workers for dataloader",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--execution_mode",
        type=str,
        choices=["pretrain", "finetune"],
        default="pretrain",
    )

    parser.add_argument(
        "--finetuning_model_ckpt",
        help="pretrained model source for fine-tuning: Lightning checkpoint, Transformers directory, or Hugging Face model ID",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--weight_decay", help="weight decay to use", type=float, default=0.01
    )

    parser.add_argument(
        "--beta1",
        help="exponential decay rate for first moment estimate",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--beta2",
        help="exponential decay rate for second movement estimate",
        type=float,
        default=0.999,
    )

    parser.add_argument(
        "--lr_scaling_rule",
        help="the LR will be scaled according to the rule LR = base_lr * effective_batch_size / 256. is useful for prototyping and is popular in vision SSL. effective_batch_size is based off bs * num_gpus * accumulate_grad_batches",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--optimizer", choices=["muon_adamw"], type=str, default="muon_adamw"
    )
    parser.add_argument(
        "--muon_lr",
        help="[Muon] Learning rate for matrix parameters",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--muon_momentum", help="[Muon] Nesterov momentum coefficient", type=float, default=0.95
    )
    parser.add_argument(
        "--muon_ns_steps", help="[Muon] Number of Polar Express iterations", type=int, default=5
    )
    parser.add_argument(
        "--muon_beta2",
        help="[Muon] beta2 for the second-moment estimate",
        type=float,
        default=0.95,
    )

    parser.add_argument(
        "--adamw_embedding_lr",
        help="[Muon] Absolute embedding learning rate; dmodel scaling is applied",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--adamw_vocab_to_embed_lr",
        help="[Muon] Absolute vocab_to_embed learning rate",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--adamw_scalar_lr",
        help="[Muon] Absolute learning rate for transformer scalar parameters",
        type=float,
        default=-1,
    )
    parser.add_argument(
        "--adamw_dmodel_lr_scaling",
        help="[Muon] Apply dmodel scaling to AdamW learning rates",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--muon_momentum_warmup_steps",
        help="[Muon] Steps for linear momentum warmup from 0.85; zero disables it",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--dynamic_wd",
        help="[Option 2] Enable weight decay that linearly decays to zero",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--linear_warmdown",
        help="[Option 3] Enable the ESM linear warmdown learning-rate schedule",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--warmup_ratio",
        help="[Option 3] Fraction of total steps used for warmup",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--warmdown_ratio",
        help="[Option 3] Fraction of total steps used for warmdown",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--final_lr_frac",
        help="[Option 3] Final learning rate as a fraction of peak learning rate",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--dataset_name", help="training task: dclm or esm_sft", default="dclm"
    )
    parser.add_argument(
        "--base_train_dataset",
        help="local base pretraining corpus used by the esm parquet dataloader",
        choices=["fineweb", "climbmix", "dclm", "owt"],
        default=os.environ.get(
            "BASE_TRAIN_DATASET", os.environ.get("ESM_BASE_TRAIN_DATASET", "dclm")
        ),
    )
    parser.add_argument(
        "--base_train_data_dir",
        help="optional override for the local parquet directory of --base_train_dataset",
        default=os.environ.get("BASE_TRAIN_DATA_DIR", ""),
    )

    parser.add_argument("--dataset_dir", help="dataset base directory", default="")

    parser.add_argument(
        "--no_randomness_dataloader",
        help="makes dataloader have no randomness by only sampling from start",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--limit_train_batches",
        help="percent of training dataset to use, or if > 1 is num batches",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--limit_val_batches",
        help="percent of validation dataset to use, or if > 1 is num batches",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--val_sanity", help="number of sanity validation steps", type=int, default=0
    )

    parser.add_argument(
        "--val_check_interval",
        help="interval to do validation per training epoch, 1 means once per epoch. useful if epochs are too large to wait to do validation",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--val_steps", help="how many steps of validation", type=int, default=1000
    )

    parser.add_argument(
        "--no_wandb", help="no wandb", action="store_true", default=False
    )
    parser.add_argument(
        "--disable_wandb",
        help="alias for --no_wandb, completely disables wandb",
        action="store_true",
        default=False,
    )

    parser.add_argument("--wandb_entity", help="wandb entity", default="")

    parser.add_argument("--wandb_project", help="wandb project name", default="ESM")

    parser.add_argument(
        "--wandb_tags", help="wandb tags to add", nargs="+", default=None
    )

    parser.add_argument(
        "--wandb_offline",
        help="set wandb to offline mode",
        action="store_true",
        default=True,
    )

    parser.add_argument(
        "--wandb_watch",
        help="turns on watch mode for wandb - expensive so only use for debugging",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--wandb_watch_level",
        help="wandb watch log level: 'parameters' (default when --wandb_watch is on), 'gradients', or 'all' (most expensive)",
        type=str,
        default="parameters",
        choices=["parameters", "gradients", "all"],
    )

    parser.add_argument(
        "--wandb_watch_log_freq",
        help="number of steps to log for wandb watch. is higher since is a bit expensive",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--console_log_filename",
        help="filename of log, used when no_wandb is active",
        default="console.log",
    )

    parser.add_argument(
        "--log_model_archi",
        help="log model architecture",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--log_gradients",
        help="logs gradients at every step to wandb to debug them",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--log_every_n_steps",
        help="turns on logger freq via pl, not advised to use this",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--resume_training_ckpt",
        help="checkpoint to resume training from, use absolute",
        type=str,
        default="",
    )

    parser.add_argument(
        "--resume_warmup_steps",
        type=int,
        default=0,
        help="Warmup steps after resume; zero disables resume warmup",
    )

    parser.add_argument(
        "--checkpoint_monitor_string",
        help="string to use to monitor for saving checkpoint. supported by PL callback",
        type=str,
        default="valid_loss",
    )

    parser.add_argument(
        "--checkpoint_monitor_mode",
        help="monitoring mode for checkpoint_monitor_string, either ['min', 'max']. if is loss do min, if is a metric like accuracy do max",
        type=str,
        default="min",
    )

    parser.add_argument(
        "--save_top_k_ckpts",
        help="number of ckpts to save when doing val (saves the ones with best metrics using checkpoint monitor string and mode defined). -1 means save all",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="",
        help="Override checkpoint directory (default: output_root/outputs/run_name/checkpoints)",
    )

    parser.add_argument(
        "--wandb_save_dir",
        type=str,
        default="",
        help="Override WandB save directory (default: run log directory/wandb)",
    )

    parser.add_argument(
        "--save_periodic_steps",
        type=int,
        default=0,
        help="Save checkpoint every N training steps regardless of val_loss (0=disabled). Useful for SFT where val_loss may rise while task performance improves.",
    )

    parser.add_argument(
        "--checkpoint_min_free_gb",
        type=float,
        default=10.0,
        help="Minimum free disk space in GiB to reserve before saving a checkpoint.",
    )

    parser.add_argument(
        "--checkpoint_preflight_required",
        action="store_true",
        default=False,
        help="Before training, fail if the checkpoint directory cannot hold the configured checkpoint count plus the free-space reserve.",
    )

    parser.add_argument(
        "--disable_checkpoint_cleanup_on_low_space",
        action="store_true",
        default=False,
        help="Disable automatic deletion of old checkpoints when free disk space is below --checkpoint_min_free_gb.",
    )

    parser.add_argument(
        "--set_matmul_precision",
        help='set math mult precision - "medium", "high", or "highest" ',
        default=None,
    )

    parser.add_argument(
        "--float_precision",
        help="float precision, pl recommends 16-mixed/bf16-mixed, also has by default 32-true",
        type=str,
        default="32-true",
    )

    parser.add_argument(
        "--compile_model",
        help="compiles the model using torch.compile",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compile_mode",
        help="torch.compile mode: full, transformer_only, or disabled",
        type=str,
        default="transformer_only",
        choices=["full", "transformer_only", "disabled"],
    )
    parser.add_argument(
        "--compile_backend",
        help="torch.compile backend: inductor, eager, or aot_eager",
        type=str,
        default="inductor",
    )
    parser.add_argument(
        "--compile_dynamic",
        help="Allow dynamic shapes, which may reduce performance",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--use_deepspeed_stage3",
        help="Enable DeepSpeed ZeRO Stage 3 with CPU offload",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gradient_checkpointing",
        help="Enable gradient checkpointing to reduce GPU memory use",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--use_sdpa_attention",
        help="Use the PyTorch SDPA attention path to reduce Path B memory use",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--is_slurm_run",
        help="please set to true if doing slurm run, as of now just stops capturing console logs",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--override_slurm_checks",
        help="dont use slurm checks to assert that certain conditions are true (i.e. train/test limit_batches is default value)",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--debug_mode",
        help="turns debug mode on where dataset returned is very small, no_wandb is on and detect anomaly is on",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--fast_dev_run",
        help="turns fast_dev_run for trainer on, makes it just do one training epoch and one val epoch",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--overfit_batches",
        help="if nonzero will overfit to specified num/percent of batches",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--profiler", choices=["simple", "advanced", "pytorch"], type=str, default=""
    )

    parser.add_argument(
        "--no_shuffle",
        help="stops shuffling - helpful for debugging",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--detect_anomaly",
        help="turns on anomaly detection mode",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--find_unused_parameters",
        help="turns on pl find unused params mode - DO NOT KEEP ON for actual training, helpful if want to debug. this uses DDPStrategy and ignores distributed_strategy",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--debug_unused_parameters",
        help="makes it so it tracks which params are used to find the params that are causing the unused params issue. need to do some things in base_model_trainer so ctrl f this hparam to see the NOTEs",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--manual_gc_collect_every_n_steps",
        help="manually call gc collect every n steps, can be done to prevent CPU RAM memory 'leak'",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--print_config_only",
        action="store_true",
        default=False,
        help="print the merged defaults/config/CLI values and exit",
    )

    args = parse_with_config(parser, kind="train")
    print_resolved_config(args, kind="train")
    if args.print_config_only:
        raise SystemExit(0)
    main(args)
