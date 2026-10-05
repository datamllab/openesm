"""
The base/pretraining dataset is a set of parquet files.
This file contains utilities for:
- iterating over the parquet files and yielding documents from it
- download the files on demand if they are not on disk

For details of how the dataset was prepared, see `repackage_data_reference.py`.
"""

import os
import argparse
import time
import requests
import pyarrow.parquet as pq
from multiprocessing import Pool

from esm.common import get_base_dir


BASE_URL = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
MAX_SHARD = 1822


def index_to_filename(index):
    return f"shard_{index:05d}.parquet"


VALID_BASE_TRAIN_DATASETS = ("fineweb", "climbmix", "dclm", "owt")
OWT_VALIDATION_DOCS = 100_000
DEFAULT_BASE_TRAIN_DATASET = "dclm"


def normalize_base_train_dataset(dataset_name=None):
    """Return a validated base pretraining dataset name."""
    dataset_name = (
        dataset_name
        or os.environ.get("BASE_TRAIN_DATASET")
        or os.environ.get("ESM_BASE_TRAIN_DATASET")
        or DEFAULT_BASE_TRAIN_DATASET
    )
    dataset_name = str(dataset_name).strip().lower()
    if dataset_name not in VALID_BASE_TRAIN_DATASETS:
        valid = ", ".join(VALID_BASE_TRAIN_DATASETS)
        raise ValueError(
            f"Unknown base_train_dataset={dataset_name!r}; expected one of: {valid}"
        )
    return dataset_name


def get_dataset_dir(dataset_name=None, data_dir=None):
    """
    Return the local parquet directory for a base pretraining dataset.

    The layout is:
      $ESM_BASE_DIR/fineweb/
      $ESM_BASE_DIR/climbmix/
      $ESM_BASE_DIR/dclm/
      $ESM_BASE_DIR/owt/
    """
    if data_dir:
        return os.path.abspath(data_dir)

    dataset_name = normalize_base_train_dataset(dataset_name)
    return os.path.join(get_base_dir(), dataset_name)


def list_parquet_files(data_dir=None, dataset_name=None):
    """Looks into a data dir and returns full paths to all parquet files."""
    data_dir = get_dataset_dir(dataset_name=dataset_name, data_dir=data_dir)
    if not os.path.isdir(data_dir):
        return []
    parquet_files = sorted(
        [
            f
            for f in os.listdir(data_dir)
            if f.endswith(".parquet") and not f.endswith(".tmp")
        ]
    )
    parquet_paths = [os.path.join(data_dir, f) for f in parquet_files]
    return parquet_paths


def get_owt_split_ranges(parquet_paths, validation_docs=OWT_VALIDATION_DOCS):
    """Return exact document ranges for the OWT train/validation split.

    The MDLM OWT protocol reserves the *last* 100,000 documents, rather than
    reserving the last parquet shard.  The returned list contains one
    ``(path, start_row, end_row)`` tuple per file that contributes to the
    split; row offsets are relative to that parquet file and ``end_row`` is
    exclusive.  Metadata is used, so no document data is read while building
    the split.
    """
    if not parquet_paths:
        return {"train": [], "val": [], "total_docs": 0}

    file_rows = [
        (path, int(pq.ParquetFile(path).metadata.num_rows)) for path in parquet_paths
    ]
    total_docs = sum(num_rows for _, num_rows in file_rows)
    if total_docs < validation_docs:
        raise ValueError(
            f"OWT contains {total_docs} documents, fewer than the requested "
            f"{validation_docs}-document validation split"
        )

    train_end = total_docs - validation_docs
    result = {"train": [], "val": [], "total_docs": total_docs}
    corpus_start = 0
    for path, num_rows in file_rows:
        corpus_end = corpus_start + num_rows
        train_start = 0
        train_stop = max(0, min(num_rows, train_end - corpus_start))
        val_start = max(0, min(num_rows, train_end - corpus_start))
        val_stop = num_rows
        if train_stop > train_start:
            result["train"].append((path, train_start, train_stop))
        if val_stop > val_start:
            result["val"].append((path, val_start, val_stop))
        corpus_start = corpus_end
    return result


def parquets_iter_batched(split, start=0, step=1, dataset_name=None, data_dir=None):
    """
    Iterate through the dataset, in batches of underlying row_groups for efficiency.
    - split can be "train" or "val". Existing datasets use the last parquet
      file for val; OWT uses an exact 100K-document boundary.
    - start/step are useful for skipping rows in DDP. e.g. start=rank, step=world_size

    Existing datasets use the historical last-shard split. OWT instead uses
    an exact document boundary at the final 100,000 documents.
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"
    parquet_paths = list_parquet_files(data_dir=data_dir, dataset_name=dataset_name)
    dataset_name = normalize_base_train_dataset(dataset_name)
    if dataset_name == "owt":
        parquet_specs = get_owt_split_ranges(parquet_paths)[split]
    else:
        selected_paths = parquet_paths[:-1] if split == "train" else parquet_paths[-1:]
        parquet_specs = [(path, 0, None) for path in selected_paths]
    for filepath, file_start, file_end in parquet_specs:
        pf = pq.ParquetFile(filepath)
        for rg_idx in range(start, pf.num_row_groups, step):
            rg = pf.read_row_group(rg_idx)
            texts = rg.column("text").to_pylist()
            if file_end is not None:
                rg_start = sum(pf.metadata.row_group(i).num_rows for i in range(rg_idx))
                rg_end = rg_start + len(texts)
                selected_start = max(rg_start, file_start)
                selected_end = min(rg_end, file_end)
                if selected_end <= selected_start:
                    continue
                texts = texts[selected_start - rg_start : selected_end - rg_start]
            yield texts


def download_single_file(index):
    """Downloads a single file index, with some backoff"""

    filename = index_to_filename(index)
    data_dir = get_dataset_dir(DEFAULT_BASE_TRAIN_DATASET)
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            temp_path = filepath + ".tmp"
            with open(temp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True

        except (requests.RequestException, IOError) as e:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {e}")

            for path in [filepath + ".tmp", filepath]:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

            if attempt < max_attempts:
                wait_time = 2**attempt
                print(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed to download {filename} after {max_attempts} attempts")
                return False

    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu 100BT dataset shards"
    )
    parser.add_argument(
        "-n",
        "--num-files",
        type=int,
        default=-1,
        help="Number of shards to download (default: -1), -1 = disable",
    )
    parser.add_argument(
        "-w",
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel download workers (default: 4)",
    )
    args = parser.parse_args()

    num = MAX_SHARD + 1 if args.num_files == -1 else min(args.num_files, MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(
        f"Downloading {len(ids_to_download)} shards using {args.num_workers} workers..."
    )
    data_dir = get_dataset_dir(DEFAULT_BASE_TRAIN_DATASET)
    print(f"Target directory: {data_dir}")
    print()
    with Pool(processes=args.num_workers) as pool:
        results = pool.map(download_single_file, ids_to_download)

    successful = sum(1 for success in results if success)
    print(f"Done! Downloaded: {successful}/{len(ids_to_download)} shards to {data_dir}")
