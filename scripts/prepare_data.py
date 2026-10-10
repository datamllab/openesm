#!/usr/bin/env python3
"""
Prepare or extend a token-budgeted local base pretraining corpus for ESM.

The output format intentionally matches esm's FineWeb parquet layout:
  <repo>/data/<dataset>/shard_00000.parquet
  <repo>/data/<dataset>/shard_00001.parquet
  ...

Each parquet file contains a single "text" column. The existing esm
dataloader uses all sorted shards except the last for train, and the last shard
for validation.
"""

from __future__ import annotations

import argparse
import fnmatch
import gzip
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from copy import deepcopy
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_TARGET_TOKENS = 7_340_032_000
DEFAULT_MARGIN = 0.05
TEXT_COLUMNS = ("text", "content", "raw_content", "document", "doc")


@dataclass(frozen=True)
class RepoSpec:
    repo_id: str
    patterns: tuple[str, ...]
    tokenized_gpt2: bool = False


DATASET_REPOS: dict[str, tuple[RepoSpec, ...]] = {
    "climbmix": (
        RepoSpec(
            repo_id="OptimalScale/ClimbMix",
            patterns=(
                "*.parquet",
                "*.jsonl",
                "*.jsonl.gz",
                "*.jsonl.zst",
                "*.jsonl.zstd",
            ),
        ),
        RepoSpec(
            repo_id="nvidia/Nemotron-ClimbMix",
            patterns=(
                "*.parquet",
                "*.jsonl",
                "*.jsonl.gz",
                "*.jsonl.zst",
                "*.jsonl.zstd",
            ),
        ),
        RepoSpec(
            repo_id="nvidia/Nemotron-ClimbMix",
            patterns=(
                "*tokenized*.jsonl",
                "*tokenized*.jsonl.gz",
                "*tokenized*.jsonl.zst",
                "*tokenized*.jsonl.zstd",
            ),
            tokenized_gpt2=True,
        ),
    ),
    "dclm": (
        RepoSpec(
            repo_id="mlfoundations/dclm-baseline-1.0-parquet",
            patterns=("*.parquet",),
        ),
        RepoSpec(
            repo_id="mlfoundations/dclm-baseline-1.0",
            patterns=("*.jsonl.zst", "*.jsonl.zstd", "*.jsonl.gz", "*.jsonl"),
        ),
    ),
}


class ParquetShardWriter:
    def __init__(
        self,
        out_dir: Path,
        tokens_per_shard: int,
        row_group_size: int,
        compression: str,
    ):
        self.out_dir = out_dir
        self.tokens_per_shard = tokens_per_shard
        self.row_group_size = row_group_size
        self.compression = compression
        self.schema = pa.schema([("text", pa.string())])
        self.writer: pq.ParquetWriter | None = None
        self.shard_index = 0
        self.shard_tokens = 0
        self.total_docs = 0
        self.shard_token_counts: list[int] = []

    def _open(self) -> None:
        if self.writer is not None:
            return
        path = self.out_dir / f"shard_{self.shard_index:05d}.parquet"
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        self.writer = pq.ParquetWriter(
            tmp_path, self.schema, compression=self.compression
        )

    def _close(self) -> None:
        if self.writer is None:
            return
        self.writer.close()
        tmp_path = self.out_dir / f"shard_{self.shard_index:05d}.parquet.tmp"
        path = self.out_dir / f"shard_{self.shard_index:05d}.parquet"
        tmp_path.replace(path)
        self.shard_token_counts.append(self.shard_tokens)
        self.writer = None
        self.shard_index += 1
        self.shard_tokens = 0

    def write(self, rows: list[str], token_count: int) -> None:
        if not rows:
            return
        self._open()
        assert self.writer is not None
        table = pa.table({"text": rows}, schema=self.schema)
        self.writer.write_table(table, row_group_size=self.row_group_size)
        self.shard_tokens += token_count
        self.total_docs += len(rows)
        if self.shard_tokens >= self.tokens_per_shard:
            self._close()

    def close(self) -> None:
        self._close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=("climbmix", "dclm", "fineweb"),
        default=["climbmix", "dclm", "fineweb"],
    )
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--tokens-per-output-shard", type=int, default=20_000_000)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--keep-sources", action="store_true")
    parser.add_argument("--allow-tokenized-climbmix", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument(
        "--max-input-files",
        type=int,
        default=0,
        help="0 means no limit; useful for smoke tests",
    )
    parser.add_argument(
        "--fineweb-shards",
        type=int,
        default=0,
        help="FineWeb shards to fetch (0 = esm.dataset default); the last one is validation",
    )
    parser.add_argument(
        "--fineweb-workers", type=int, default=4, help="parallel FineWeb downloads"
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), message, flush=True)


def natural_key(path: str) -> list[object]:
    parts: list[object] = []
    buf = ""
    for ch in path:
        if ch.isdigit():
            buf += ch
        else:
            if buf:
                parts.append(int(buf))
                buf = ""
            parts.append(ch)
    if buf:
        parts.append(int(buf))
    return parts


def bytes_in_tree(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for item in path.rglob("*"):
        if item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024**3):.2f} GiB"


def check_free_space(path: Path, min_free_gb: float) -> None:
    usage = shutil.disk_usage(path)
    free_gb = usage.free / (1024**3)
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"Free space under {path} is {free_gb:.2f} GiB, below --min-free-gb={min_free_gb}"
        )


def load_tokenizer(repo_root: Path):
    from esm.config import bootstrap_assets

    bootstrap_assets("train", fallback=repo_root / "data")
    from esm.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    bos = tokenizer.get_bos_token_id()
    log(f"Loaded tokenizer from {os.environ['ESM_BASE_DIR']}/tokenizer (bos={bos})")
    return tokenizer, bos


def matching_files(api: HfApi, spec: RepoSpec) -> list[str]:
    log(f"Listing dataset repo {spec.repo_id}")
    files = api.list_repo_files(spec.repo_id, repo_type="dataset")
    matched = [
        name
        for name in files
        if not name.endswith("/")
        and any(fnmatch.fnmatch(name, pattern) for pattern in spec.patterns)
        and not Path(name).name.startswith(".")
    ]
    matched = sorted(set(matched), key=natural_key)
    log(f"Matched {len(matched)} candidate file(s) in {spec.repo_id}")
    return matched


def download_file(
    spec: RepoSpec, remote_path: str, local_dir: Path, hf_cache_dir: Path
) -> Path:
    repo_local_dir = local_dir / spec.repo_id.replace("/", "__")
    repo_local_dir.mkdir(parents=True, exist_ok=True)
    log(f"Downloading {spec.repo_id}:{remote_path}")
    try:
        path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=remote_path,
            repo_type="dataset",
            local_dir=repo_local_dir,
            cache_dir=hf_cache_dir,
            local_dir_use_symlinks=False,
        )
    except TypeError:
        path = hf_hub_download(
            repo_id=spec.repo_id,
            filename=remote_path,
            repo_type="dataset",
            local_dir=repo_local_dir,
            cache_dir=hf_cache_dir,
        )
    return Path(path)


def open_text_lines(path: Path) -> Iterator[str]:
    name = path.name
    if name.endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            yield from handle
    elif name.endswith(".zst") or name.endswith(".zstd"):
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError(
                "Reading zstd jsonl requires the Python package 'zstandard'. "
                "Prefer the parquet mirror or install zstandard on the cpu-worker."
            ) from exc
        with path.open("rb") as raw:
            reader = zstd.ZstdDecompressor().stream_reader(raw)
            text_stream = getattr(reader, "read", None)
            if text_stream is None:
                raise RuntimeError("Could not create zstd stream reader")
            import io

            with io.TextIOWrapper(reader, encoding="utf-8", errors="replace") as handle:
                yield from handle
    else:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            yield from handle


def extract_json_text(obj: object) -> str | None:
    if isinstance(obj, str):
        return obj
    if not isinstance(obj, dict):
        return None
    for key in TEXT_COLUMNS:
        value = obj.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def detokenize_gpt2_json(obj: object) -> str | None:
    if not isinstance(obj, dict):
        return None
    token_ids = None
    for key in ("tokens", "token_ids", "input_ids", "ids"):
        value = obj.get(key)
        if (
            isinstance(value, list)
            and value
            and all(isinstance(item, int) for item in value)
        ):
            token_ids = value
            break
    if token_ids is None:
        return extract_json_text(obj)
    import tiktoken

    return tiktoken.get_encoding("gpt2").decode(token_ids)


def iter_jsonl_texts(path: Path, tokenized_gpt2: bool) -> Iterator[str]:
    for line_number, line in enumerate(open_text_lines(path), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            if line_number <= 5:
                log(f"Skipping invalid JSON in {path.name}:{line_number}")
            continue
        text = detokenize_gpt2_json(obj) if tokenized_gpt2 else extract_json_text(obj)
        if text:
            yield text


def parquet_text_column(path: Path) -> str:
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    for column in TEXT_COLUMNS:
        if column in names:
            return column
    raise RuntimeError(f"No text-like column found in {path}; columns={names}")


def iter_parquet_texts(path: Path) -> Iterator[str]:
    pf = pq.ParquetFile(path)
    column = parquet_text_column(path)
    for rg_idx in range(pf.num_row_groups):
        table = pf.read_row_group(rg_idx, columns=[column])
        for value in table.column(column).to_pylist():
            if isinstance(value, str) and value:
                yield value


def iter_texts(path: Path, tokenized_gpt2: bool) -> Iterator[str]:
    if path.name.endswith(".parquet"):
        yield from iter_parquet_texts(path)
    else:
        yield from iter_jsonl_texts(path, tokenized_gpt2=tokenized_gpt2)


def maybe_remove_source(path: Path, sources_dir: Path, keep_sources: bool) -> None:
    if keep_sources:
        return
    try:
        path.relative_to(sources_dir)
    except ValueError:
        return
    if path.exists() and path.is_file():
        path.unlink()


def write_manifest(out_dir: Path, manifest: dict) -> None:
    tmp = out_dir / "manifest.json.tmp"
    final = out_dir / "manifest.json"
    tmp.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    tmp.replace(final)


def prepare_dataset(
    args: argparse.Namespace, dataset_name: str, tokenizer, bos: int
) -> None:
    repo_root = args.repo_root.resolve()
    data_root = repo_root / "data"
    out_dir = data_root / dataset_name
    sources_dir = out_dir / "_sources"
    hf_cache_dir = args.hf_cache_dir or (data_root / "hf_cache")
    target_with_margin = int(args.target_tokens * (1.0 + args.margin))

    out_dir.mkdir(parents=True, exist_ok=True)
    sources_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest.get("complete")
            and int(manifest.get("total_tokens", 0)) >= args.target_tokens
        ):
            log(
                f"{dataset_name}: existing complete manifest has {manifest['total_tokens']:,} tokens; skipping"
            )
            return
        raise RuntimeError(
            f"{dataset_name}: incomplete existing manifest found at {manifest_path}; rerun with --force"
        )

    if args.force:
        for old_file in out_dir.glob("shard_*.parquet"):
            old_file.unlink()
        if manifest_path.exists():
            manifest_path.unlink()

    api = HfApi()
    writer = ParquetShardWriter(
        out_dir=out_dir,
        tokens_per_shard=args.tokens_per_output_shard,
        row_group_size=args.row_group_size,
        compression=args.parquet_compression,
    )
    manifest: dict = {
        "dataset": dataset_name,
        "target_tokens": args.target_tokens,
        "target_tokens_with_margin": target_with_margin,
        "total_tokens": 0,
        "total_docs": 0,
        "complete": False,
        "files": [],
        "output_dir": str(out_dir),
    }

    total_tokens = 0
    input_files_seen = 0
    selected_specs = DATASET_REPOS[dataset_name]
    try:
        for spec in selected_specs:
            if spec.tokenized_gpt2 and not args.allow_tokenized_climbmix:
                log(
                    f"Skipping tokenized fallback {spec.repo_id}; pass --allow-tokenized-climbmix to enable it"
                )
                continue
            remote_files = matching_files(api, spec)
            if not remote_files:
                continue
            for remote_path in remote_files:
                if args.max_input_files and input_files_seen >= args.max_input_files:
                    log(
                        f"{dataset_name}: stopping early because --max-input-files={args.max_input_files}"
                    )
                    break
                check_free_space(data_root, args.min_free_gb)
                local_path = download_file(spec, remote_path, sources_dir, hf_cache_dir)
                input_files_seen += 1

                file_tokens = 0
                file_docs = 0
                pending_rows: list[str] = []
                pending_tokens = 0
                for text in iter_texts(local_path, tokenized_gpt2=spec.tokenized_gpt2):
                    token_count = len(tokenizer.encode(text, prepend=bos))
                    if token_count <= 1:
                        continue
                    pending_rows.append(text)
                    pending_tokens += token_count
                    file_tokens += token_count
                    file_docs += 1
                    total_tokens += token_count
                    if len(pending_rows) >= args.row_group_size:
                        writer.write(pending_rows, pending_tokens)
                        pending_rows = []
                        pending_tokens = 0
                    if total_tokens >= target_with_margin:
                        break
                writer.write(pending_rows, pending_tokens)
                maybe_remove_source(
                    local_path, sources_dir, keep_sources=args.keep_sources
                )

                manifest["files"].append(
                    {
                        "repo_id": spec.repo_id,
                        "path": remote_path,
                        "docs": file_docs,
                        "tokens": file_tokens,
                    }
                )
                manifest["total_tokens"] = total_tokens
                manifest["total_docs"] = writer.total_docs
                manifest["shards_written"] = writer.shard_index + (
                    1 if writer.writer is not None else 0
                )
                write_manifest(out_dir, manifest)
                log(
                    f"{dataset_name}: processed {spec.repo_id}:{remote_path} "
                    f"docs={file_docs:,} file_tokens={file_tokens:,} total_tokens={total_tokens:,}"
                )
                if total_tokens >= target_with_margin:
                    break
            if total_tokens >= target_with_margin:
                break
        writer.close()
    except Exception:
        writer.close()
        raise

    if total_tokens < args.target_tokens:
        raise RuntimeError(
            f"{dataset_name}: stopped after {total_tokens:,} tokens, below target {args.target_tokens:,}. "
            "Check repo availability or increase --max-input-files."
        )

    manifest["total_tokens"] = total_tokens
    manifest["total_docs"] = writer.total_docs
    manifest["shards_written"] = writer.shard_index
    manifest["shard_token_counts"] = writer.shard_token_counts
    manifest["complete"] = True
    manifest["disk_bytes"] = bytes_in_tree(out_dir)
    write_manifest(out_dir, manifest)
    log(
        f"{dataset_name}: complete tokens={total_tokens:,} docs={writer.total_docs:,} "
        f"shards={writer.shard_index:,} disk={format_gib(manifest['disk_bytes'])}"
    )


def parse_append_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--append", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=tuple(DATASET_REPOS), default="dclm")
    parser.add_argument("--target-tokens", type=int, default=DEFAULT_TARGET_TOKENS)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--tokens-per-output-shard", type=int, default=20_000_000)
    parser.add_argument("--row-group-size", type=int, default=1024)
    parser.add_argument("--parquet-compression", default="zstd")
    parser.add_argument("--hf-cache-dir", type=Path, default=None)
    parser.add_argument("--keep-sources", action="store_true")
    parser.add_argument("--allow-tokenized-climbmix", action="store_true")
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument(
        "--max-input-files",
        type=int,
        default=0,
        help="0 means no limit; useful for smoke tests",
    )
    return parser.parse_args()


def load_manifest(manifest_path: Path) -> dict:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing existing manifest: {manifest_path}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def validate_existing_dataset(
    out_dir: Path, dataset_name: str, manifest: dict
) -> tuple[int, int, list[int]]:
    if manifest.get("dataset") != dataset_name:
        raise RuntimeError(
            f"Manifest dataset={manifest.get('dataset')!r}, expected {dataset_name!r}"
        )
    if not manifest.get("complete"):
        raise RuntimeError(
            "Existing manifest is not complete; refusing to append to a partial dataset"
        )
    if not manifest.get("files"):
        raise RuntimeError("Existing manifest has no input file history")

    shard_token_counts = [
        int(value) for value in manifest.get("shard_token_counts", [])
    ]
    if not shard_token_counts:
        raise RuntimeError("Existing manifest has no shard_token_counts")

    shards_written = int(manifest.get("shards_written", len(shard_token_counts)))
    if shards_written != len(shard_token_counts):
        raise RuntimeError(
            f"Manifest shards_written={shards_written}, but len(shard_token_counts)={len(shard_token_counts)}"
        )

    total_tokens = int(manifest.get("total_tokens", 0))
    if sum(shard_token_counts) != total_tokens:
        raise RuntimeError(
            f"Manifest token mismatch: sum(shard_token_counts)={sum(shard_token_counts)}, "
            f"total_tokens={total_tokens}"
        )

    for index in range(shards_written):
        shard_path = out_dir / f"shard_{index:05d}.parquet"
        if not shard_path.exists():
            raise FileNotFoundError(f"Missing existing output shard: {shard_path}")

    next_shard_path = out_dir / f"shard_{shards_written:05d}.parquet"
    if next_shard_path.exists():
        raise RuntimeError(f"Next output shard already exists: {next_shard_path}")

    return total_tokens, int(manifest.get("total_docs", 0)), shard_token_counts


def selected_specs(dataset_name: str, allow_tokenized_climbmix: bool):
    for spec in DATASET_REPOS[dataset_name]:
        if spec.tokenized_gpt2 and not allow_tokenized_climbmix:
            log(
                f"Skipping tokenized fallback {spec.repo_id}; pass --allow-tokenized-climbmix to enable it"
            )
            continue
        yield spec


def process_remote_file(
    *,
    writer: ParquetShardWriter,
    tokenizer,
    bos: int,
    local_path: Path,
    tokenized_gpt2: bool,
    skip_valid_docs: int,
    total_tokens: int,
    target_with_margin: int,
    row_group_size: int,
) -> tuple[int, int, int, bool]:
    added_docs = 0
    added_tokens = 0
    skipped_docs = 0
    pending_rows: list[str] = []
    pending_tokens = 0
    reached_target = False

    for text in iter_texts(local_path, tokenized_gpt2=tokenized_gpt2):
        token_count = len(tokenizer.encode(text, prepend=bos))
        if token_count <= 1:
            continue
        if skipped_docs < skip_valid_docs:
            skipped_docs += 1
            continue

        pending_rows.append(text)
        pending_tokens += token_count
        added_docs += 1
        added_tokens += token_count
        total_tokens += token_count

        if len(pending_rows) >= row_group_size:
            writer.write(pending_rows, pending_tokens)
            pending_rows = []
            pending_tokens = 0

        if total_tokens >= target_with_margin:
            reached_target = True
            break

    if skipped_docs < skip_valid_docs:
        raise RuntimeError(
            f"Existing manifest says {skip_valid_docs} valid docs were consumed from {local_path.name}, "
            f"but only {skipped_docs} could be skipped"
        )

    writer.write(pending_rows, pending_tokens)
    return added_docs, added_tokens, total_tokens, reached_target


def move_appended_shards(staging_dir: Path, out_dir: Path) -> None:
    for shard_path in sorted(staging_dir.glob("shard_*.parquet")):
        target_path = out_dir / shard_path.name
        if target_path.exists():
            raise RuntimeError(f"Refusing to overwrite existing shard: {target_path}")
        shard_path.replace(target_path)
    staging_dir.rmdir()


def append_dataset(args: argparse.Namespace, tokenizer, bos: int) -> None:
    repo_root = args.repo_root.resolve()
    data_root = repo_root / "data"
    out_dir = data_root / args.dataset
    sources_dir = out_dir / "_sources"
    hf_cache_dir = args.hf_cache_dir or (data_root / "hf_cache")
    target_with_margin = int(args.target_tokens * (1.0 + args.margin))
    manifest_path = out_dir / "manifest.json"

    manifest = load_manifest(manifest_path)
    existing_tokens, existing_docs, shard_token_counts = validate_existing_dataset(
        out_dir, args.dataset, manifest
    )

    if existing_tokens >= target_with_margin:
        log(
            f"{args.dataset}: existing tokens={existing_tokens:,} already satisfy "
            f"target_with_margin={target_with_margin:,}; nothing to append"
        )
        return

    sources_dir.mkdir(parents=True, exist_ok=True)
    hf_cache_dir.mkdir(parents=True, exist_ok=True)
    check_free_space(data_root, args.min_free_gb)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    staging_dir = out_dir / f"_append_staging_{timestamp}"
    if staging_dir.exists():
        raise RuntimeError(f"Append staging directory already exists: {staging_dir}")
    staging_dir.mkdir(parents=True)

    writer = ParquetShardWriter(
        out_dir=staging_dir,
        tokens_per_shard=args.tokens_per_output_shard,
        row_group_size=args.row_group_size,
        compression=args.parquet_compression,
    )
    writer.shard_index = len(shard_token_counts)

    new_manifest = deepcopy(manifest)
    new_manifest["complete"] = False
    new_manifest["target_tokens"] = args.target_tokens
    new_manifest["target_tokens_with_margin"] = target_with_margin
    manifest_files = new_manifest["files"]
    last_file = manifest_files[-1]
    start_repo_id = last_file["repo_id"]
    start_path = last_file["path"]
    skip_docs_for_start = int(last_file.get("docs", 0))

    total_tokens = existing_tokens
    total_docs = existing_docs
    api = HfApi()
    found_start = False
    downloaded_inputs = 0

    log(
        f"{args.dataset}: append from {existing_tokens:,} tokens to target={args.target_tokens:,} "
        f"(effective_stop={target_with_margin:,})"
    )
    log(
        f"{args.dataset}: resuming at {start_repo_id}:{start_path} after {skip_docs_for_start:,} valid docs"
    )

    try:
        for spec in selected_specs(args.dataset, args.allow_tokenized_climbmix):
            remote_files = matching_files(api, spec)
            if not remote_files:
                continue

            start_index = 0
            if not found_start:
                if spec.repo_id != start_repo_id:
                    continue
                try:
                    start_index = remote_files.index(start_path)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Start path is not present in remote file list: {start_path}"
                    ) from exc
                found_start = True

            for remote_path in remote_files[start_index:]:
                if args.max_input_files and downloaded_inputs >= args.max_input_files:
                    log(
                        f"{args.dataset}: stopping early because --max-input-files={args.max_input_files}"
                    )
                    break

                check_free_space(data_root, args.min_free_gb)
                local_path = download_file(spec, remote_path, sources_dir, hf_cache_dir)
                downloaded_inputs += 1

                skip_docs = (
                    skip_docs_for_start
                    if spec.repo_id == start_repo_id and remote_path == start_path
                    else 0
                )
                added_docs, added_tokens, total_tokens, reached_target = (
                    process_remote_file(
                        writer=writer,
                        tokenizer=tokenizer,
                        bos=bos,
                        local_path=local_path,
                        tokenized_gpt2=spec.tokenized_gpt2,
                        skip_valid_docs=skip_docs,
                        total_tokens=total_tokens,
                        target_with_margin=target_with_margin,
                        row_group_size=args.row_group_size,
                    )
                )
                total_docs += added_docs
                maybe_remove_source(
                    local_path, sources_dir, keep_sources=args.keep_sources
                )

                if added_docs:
                    if spec.repo_id == start_repo_id and remote_path == start_path:
                        manifest_files[-1]["docs"] = (
                            int(manifest_files[-1].get("docs", 0)) + added_docs
                        )
                        manifest_files[-1]["tokens"] = (
                            int(manifest_files[-1].get("tokens", 0)) + added_tokens
                        )
                    else:
                        manifest_files.append(
                            {
                                "repo_id": spec.repo_id,
                                "path": remote_path,
                                "docs": added_docs,
                                "tokens": added_tokens,
                            }
                        )

                log(
                    f"{args.dataset}: appended {spec.repo_id}:{remote_path} "
                    f"skip_docs={skip_docs:,} docs={added_docs:,} tokens={added_tokens:,} "
                    f"total_tokens={total_tokens:,}"
                )

                if reached_target:
                    break
            if total_tokens >= target_with_margin:
                break
            if args.max_input_files and downloaded_inputs >= args.max_input_files:
                break
        writer.close()
    except Exception:
        writer.close()
        raise

    if not found_start:
        raise RuntimeError(
            f"Could not find existing manifest start repo in DATASET_REPOS: {start_repo_id}"
        )
    if total_tokens < args.target_tokens:
        raise RuntimeError(
            f"{args.dataset}: stopped after {total_tokens:,} tokens, below target {args.target_tokens:,}. "
            f"New shards remain staged at {staging_dir}"
        )

    appended_shard_counts = writer.shard_token_counts
    backup_path = out_dir / f"manifest.before_append_{timestamp}.json"
    shutil.copy2(manifest_path, backup_path)
    move_appended_shards(staging_dir, out_dir)

    new_manifest["total_tokens"] = total_tokens
    new_manifest["total_docs"] = total_docs
    new_manifest["shards_written"] = len(shard_token_counts) + len(
        appended_shard_counts
    )
    new_manifest["shard_token_counts"] = shard_token_counts + appended_shard_counts
    new_manifest["complete"] = True
    new_manifest["disk_bytes"] = bytes_in_tree(out_dir)
    new_manifest.setdefault("append_history", []).append(
        {
            "timestamp": timestamp,
            "previous_total_tokens": existing_tokens,
            "previous_total_docs": existing_docs,
            "new_total_tokens": total_tokens,
            "new_total_docs": total_docs,
            "target_tokens": args.target_tokens,
            "target_tokens_with_margin": target_with_margin,
            "appended_tokens": total_tokens - existing_tokens,
            "appended_docs": total_docs - existing_docs,
            "appended_shards": len(appended_shard_counts),
            "manifest_backup": str(backup_path),
        }
    )
    write_manifest(out_dir, new_manifest)

    log(
        f"{args.dataset}: append complete appended_tokens={total_tokens - existing_tokens:,} "
        f"appended_shards={len(appended_shard_counts):,} total_tokens={total_tokens:,} "
        f"disk={format_gib(new_manifest['disk_bytes'])}"
    )
    log(f"{args.dataset}: previous manifest backed up at {backup_path}")


def append_main() -> None:
    args = parse_append_args()
    args.repo_root = args.repo_root.resolve()
    log(f"repo_root={args.repo_root}")
    log(f"dataset={args.dataset}")
    log(f"target_tokens={args.target_tokens:,}; margin={args.margin:.2%}")
    log(f"effective_stop_tokens={int(args.target_tokens * (1.0 + args.margin)):,}")
    tokenizer, bos = load_tokenizer(args.repo_root)
    append_dataset(args, tokenizer, bos)


def main() -> None:
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    os.environ.setdefault("HF_HOME", str(args.repo_root / "data" / "hf_home"))
    os.environ.setdefault(
        "HF_DATASETS_CACHE", str(args.repo_root / "data" / "hf_datasets_cache")
    )
    log(f"repo_root={args.repo_root}")
    log(f"datasets={args.datasets}")
    log(f"target_tokens={args.target_tokens:,}; margin={args.margin:.2%}")
    log(f"effective_stop_tokens={int(args.target_tokens * (1.0 + args.margin)):,}")
    tokenizer = bos = None
    for dataset_name in args.datasets:
        if dataset_name == "fineweb":
            # FineWeb-Edu shards are already in the target layout, so they only
            # need downloading (no repackaging or token budgeting).
            from esm.dataset import FINEWEB_DEFAULT_SHARDS, download_fineweb

            download_fineweb(
                num_shards=args.fineweb_shards or FINEWEB_DEFAULT_SHARDS,
                num_workers=args.fineweb_workers,
                data_dir=str(args.repo_root / "data" / "fineweb"),
            )
            continue
        if tokenizer is None:
            tokenizer, bos = load_tokenizer(args.repo_root)
        prepare_dataset(args, dataset_name, tokenizer, bos)
    data_root = args.repo_root / "data"
    log(f"Final data dir size: {format_gib(bytes_in_tree(data_root))} at {data_root}")


if __name__ == "__main__":
    if "--append" in sys.argv:
        append_main()
    else:
        main()
