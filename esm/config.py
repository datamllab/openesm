"""Small YAML configuration loader shared by the train/eval entry points.

The merge order is deliberately explicit:

    parser safety defaults < YAML file < explicitly supplied CLI options

This module does not define a second schema.  YAML keys are mapped to the
existing argparse destinations where their names differ, and extra keys are
kept on the Namespace so the resolved configuration can be printed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[1]
_ENV_VALUE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def default_config_path(kind: str) -> Path:
    if kind not in {"train", "eval"}:
        raise ValueError(f"unknown config kind: {kind}")
    return REPO_ROOT / "configs" / f"{kind}.yaml"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load YAML and expand portable ``${VAR:-default}`` values."""

    import yaml

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        values = yaml.safe_load(handle)
    if not isinstance(values, dict):
        raise ValueError(f"configuration root must be a YAML mapping: {config_path}")
    return _expand_environment(values)


def _expand_environment(value: Any) -> Any:
    """Expand environment placeholders without hard-coding machine paths."""

    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, fallback = match.group(1), match.group(2)
        if name in os.environ:
            return os.environ[name]
        return fallback if fallback is not None else match.group(0)

    return _ENV_VALUE.sub(replace, value)


def bootstrap_assets(kind: str, fallback: str | Path | None = None) -> dict[str, Any]:
    """Set the default ESM asset root before importing data/model modules."""

    values = load_yaml(default_config_path(kind))
    asset_root = values.get("esm_asset_root") or fallback
    if asset_root:
        os.environ.setdefault("ESM_BASE_DIR", str(asset_root))
    return values


def _explicit_destinations(
    parser: argparse.ArgumentParser, argv: Iterable[str]
) -> set[str]:
    """Return argparse destinations explicitly present in argv.

    This supports both ``--option value`` and ``--option=value``.  It is used
    after argparse has populated its normal defaults, so only explicit CLI
    values can override the YAML file.
    """

    explicit: set[str] = set()
    option_actions = parser._option_string_actions
    for token in argv:
        option = token.split("=", 1)[0] if token.startswith("-") else None
        if option is None:
            continue
        action = option_actions.get(option)
        if action is not None:
            explicit.add(action.dest)
    return explicit


_ALIASES = {
    "device_batch_size": "batch_size_per_device",
    "grad_accum": "accumulate_grad_batches",
    "save_top_k": "save_top_k_ckpts",
    "manual_gc_steps": "manual_gc_collect_every_n_steps",
    "matmul_precision": "set_matmul_precision",
    "checkpoint": "checkpoint",
    "data_root": "data_dir",
}

_TOKEN_BUDGET_ONLY_KEYS = {
    "max_steps",
    "max_scheduling_steps",
    "accumulate_grad_batches",
}


def _normalise_value(value: Any, current: Any = None) -> Any:
    """Convert YAML scalar values to values suitable for argparse attributes."""

    if (
        current is None
        and isinstance(value, str)
        and re.fullmatch(r"[+-]?\d+", value.strip())
    ):
        return int(value)
    if isinstance(current, int) and isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    if isinstance(current, float) and isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            pass
    if isinstance(current, bool) and isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    if isinstance(current, tuple):
        return tuple(value)
    return value


def resolve_path(value: str | Path, *, repo_root: Path = REPO_ROOT) -> Path:
    """Resolve a user-facing path relative to the repository root."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = repo_root / path
    return path.resolve()


def _validate_run_name(run_name: str) -> str:
    """Keep run names as single directory components."""

    candidate = str(run_name).strip()
    if (
        not candidate
        or candidate in {".", ".."}
        or "/" in candidate
        or "\\" in candidate
    ):
        raise ValueError(
            "run_name must be a non-empty single directory name without '/' or '\\': "
            f"{run_name!r}"
        )
    return candidate


def resolve_run_paths(
    args: argparse.Namespace, *, repo_root: Path = REPO_ROOT
) -> argparse.Namespace:
    """Resolve the canonical output layout for a training run.

    Relative paths are intentionally interpreted from the project root, not
    from the caller's current directory. This makes the same launcher behave
    identically when called from a local shell or from an rjob container.
    """

    output_root = resolve_path(
        getattr(args, "output_root", ".") or ".", repo_root=repo_root
    )
    run_name = getattr(args, "run_name", "") or ""
    if not run_name:
        prefix = getattr(args, "run_prefix", "pretrain-dclm-d6") or "pretrain-dclm-d6"
        if getattr(args, "execution_mode", "") == "finetune" and prefix.startswith(
            "pretrain-"
        ):
            prefix = "sft-" + prefix[len("pretrain-") :]
        run_name = f"{prefix}-{datetime.now():%Y%m%d}"
    run_name = _validate_run_name(run_name)

    run_dir = output_root / "outputs" / run_name
    log_root = (
        resolve_path(args.log_root, repo_root=repo_root)
        if getattr(args, "log_root", "")
        else run_dir / "logs"
    )
    checkpoint_dir = (
        resolve_path(args.checkpoint_dir, repo_root=repo_root)
        if getattr(args, "checkpoint_dir", "")
        else run_dir / "checkpoints"
    )
    wandb_save_dir = (
        resolve_path(args.wandb_save_dir, repo_root=repo_root)
        if getattr(args, "wandb_save_dir", "")
        else log_root / "wandb"
    )

    for path in (run_dir, log_root, checkpoint_dir, wandb_save_dir):
        path.mkdir(parents=True, exist_ok=True)

    args.output_root = str(output_root)
    args.run_name = run_name
    args.run_dir = str(run_dir)
    args.log_root = str(log_root)
    args.checkpoint_dir = str(checkpoint_dir)
    args.wandb_save_dir = str(wandb_save_dir)
    return args


def _apply_derived_values(
    args: argparse.Namespace, values: dict[str, Any], explicit: set[str]
) -> None:
    """Resolve token-budget training quantities that cannot be independently overridden."""

    if "base_train_data_dir" not in explicit and values.get("base_train_data_root"):
        dataset = values.get("base_train_dataset", values.get("dataset_name", ""))
        if dataset:
            args.base_train_data_dir = str(
                Path(values["base_train_data_root"]) / str(dataset)
            )

    if os.environ.get("WORLD_SIZE"):
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        node_count = int(os.environ.get("NODE_COUNT", values.get("node_count", 1)))
        proc_per_node = int(
            os.environ.get("PROC_PER_NODE", values.get("proc_per_node", 1))
        )
        world_size = node_count * proc_per_node

    device_batch = int(getattr(args, "batch_size_per_device", 0) or 0)
    global_batch = int(getattr(args, "global_batch_size", 0) or 0)
    if device_batch <= 0:
        raise ValueError("batch_size_per_device must be positive")
    if global_batch <= 0:
        raise ValueError("global_batch_size must be positive")

    micro_batch = world_size * device_batch
    if global_batch % micro_batch != 0:
        raise ValueError(
            "global_batch_size must be divisible by world_size × batch_size_per_device: "
            f"global_batch_size={global_batch}, world_size={world_size}, "
            f"batch_size_per_device={device_batch}"
        )
    args.accumulate_grad_batches = global_batch // micro_batch
    if args.accumulate_grad_batches <= 0:
        raise ValueError("derived accumulate_grad_batches must be positive")

    target_total_tokens = int(getattr(args, "target_total_tokens", 0) or 0)
    if target_total_tokens <= 0:
        raise ValueError(
            "TARGET_TOTAL_TOKENS must be a positive integer; set it in the environment "
            "or in the training config"
        )

    context_length = int(getattr(args, "context_length", 0) or 0)
    tokens_per_step = (
        world_size * device_batch * args.accumulate_grad_batches * context_length
    )
    if tokens_per_step <= 0:
        raise ValueError("tokens_per_step must be positive")

    max_steps = target_total_tokens // tokens_per_step
    if max_steps <= 0:
        raise ValueError(
            "TARGET_TOTAL_TOKENS is smaller than one optimizer step: "
            f"target={target_total_tokens}, tokens_per_step={tokens_per_step}"
        )

    args.max_steps = max_steps

    asset_root = values.get("esm_asset_root")
    if asset_root and "ESM_BASE_DIR" not in os.environ:
        os.environ["ESM_BASE_DIR"] = str(asset_root)

    compile_flags = values.get("compile_flags")
    if (
        compile_flags
        and "compile_model" not in explicit
        and "compile_mode" not in explicit
    ):
        flags = set(shlex.split(str(compile_flags)))
        args.compile_model = "--compile_model" in flags
        if "--compile_mode" in flags:
            index = list(shlex.split(str(compile_flags))).index("--compile_mode")
            tokens = shlex.split(str(compile_flags))
            if index + 1 < len(tokens):
                args.compile_mode = tokens[index + 1]


def merge_yaml_into_args(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    values: dict[str, Any],
    argv: Iterable[str],
    *,
    kind: str | None = None,
) -> argparse.Namespace:
    """Merge YAML values into parsed args while preserving explicit CLI values."""

    explicit = _explicit_destinations(parser, argv)
    if kind == "train":
        conflicting = sorted(
            key for key in values if _ALIASES.get(key, key) in _TOKEN_BUDGET_ONLY_KEYS
        )
        if conflicting:
            raise ValueError(
                "training length and gradient accumulation are derived from "
                "TARGET_TOTAL_TOKENS, global_batch_size, batch_size_per_device, "
                "and the distributed world size; remove these conflicting config "
                f"keys: {', '.join(conflicting)}"
            )
    for key, value in values.items():
        dest = _ALIASES.get(key, key)
        if dest in explicit:
            continue
        current = getattr(args, dest, None)
        setattr(args, dest, _normalise_value(value, current))
    if kind == "train":
        _apply_derived_values(args, values, explicit)

    if hasattr(args, "checkpoint_dir"):
        resolve_run_paths(args)
    args.config_values = dict(values)
    return args


def parse_with_config(
    parser: argparse.ArgumentParser,
    *,
    kind: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    """Parse an argparse parser with the repository's YAML precedence rules."""

    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config_path(kind)),
        help=f"YAML configuration file (default: configs/{kind}.yaml)",
    )
    args = parser.parse_args(raw_argv)
    values = load_yaml(args.config)
    return merge_yaml_into_args(parser, args, values, raw_argv, kind=kind)


def print_resolved_config(args: argparse.Namespace, *, kind: str) -> None:
    """Print the final merged configuration so it is captured in the run log."""

    payload = {
        key: value for key, value in vars(args).items() if key != "config_values"
    }
    print(f"[config] kind={kind} file={getattr(args, 'config', '')}")
    print("[config] precedence=argparse defaults < YAML < explicit CLI")
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
