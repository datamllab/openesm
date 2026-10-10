import argparse
import os
import time
from multiprocessing import Pool

import requests
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as _IterableDataset

from esm.config import REPO_ROOT
from esm.dataloader import StatefulBestFitDataLoader


class IterableDataset(_IterableDataset):
    """
    Wraps StatefulBestFitDataLoader into a PyTorch IterableDataset.

    This keeps:
    - infinite streaming
    - exact resume state support (doc_buffer + doc_batch_index)
    - no padding
    - distributed compatibility
    """

    def __init__(
        self,
        tokenizer,
        batch_size,
        max_len,
        split,
        max_iter,
        device="cuda",
        resume_state_dict=None,
        base_train_dataset="dclm",
        base_train_data_dir=None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.resume_state_dict = resume_state_dict
        self.base_train_dataset = base_train_dataset
        self.base_train_data_dir = base_train_data_dir
        self.batch_idx = 0
        self.last_state_dict = None
        self._stateful_loader = None

    def __iter__(self):
        self._stateful_loader = StatefulBestFitDataLoader(
            tokenizer=self.tokenizer,
            B=self.B,
            T=self.T,
            split=self.split,
            device=self.device,
            resume_state_dict=self.resume_state_dict,
            dataset_name=self.base_train_dataset,
            data_dir=self.base_train_data_dir,
        )
        for inputs, targets, state_dict in self._stateful_loader:
            self.last_state_dict = state_dict
            yield inputs, targets

    def get_dataloader_state(self):
        """Return exact-resume state (includes doc_buffer)."""
        if self._stateful_loader is not None:
            return self._stateful_loader.state_dict()
        return self.last_state_dict

    def __len__(self):
        return self.max_iter


def generate_dataloader(
    tokenizer,
    batch_size,
    max_len,
    max_iter,
    split,
    device,
    resume_state_dict=None,
    base_train_dataset="dclm",
    base_train_data_dir=None,
):

    dataset = IterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
        base_train_dataset=base_train_dataset,
        base_train_data_dir=base_train_data_dir,
    )

    dataloader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=False
    )
    return dataloader


# ---------------------------------------------------------------------------
# FineWeb-Edu 100BT download
#
# The base pretraining data is a set of parquet shards hosted on Hugging Face.
# Fetching is kept separate from training so it can run once on a networked
# cpu-worker. The last downloaded shard is the validation split, so the shard
# count must be at least 2. Downloading the first 370 shards (~30 GB) is enough
# for the default recipe.

FINEWEB_BASE_URL = (
    "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
)
FINEWEB_MAX_SHARD = 1822  # shard_01822.parquet is the final shard
FINEWEB_DEFAULT_SHARDS = 370


def fineweb_filename(index):
    return f"shard_{index:05d}.parquet"


def fineweb_dir(data_dir=None):
    """Directory the loader expects for ``base_train_dataset=fineweb``.

    Defaults to ``<repo>/data/fineweb`` to match ``base_train_data_root: data``
    in ``configs/train.yaml``.
    """

    if data_dir:
        return os.path.abspath(data_dir)
    return str(REPO_ROOT / "data" / "fineweb")


def _download_fineweb_shard(task):
    """Download one shard index, with backoff. Runs in a worker process."""

    index, data_dir = task
    filename = fineweb_filename(index)
    filepath = os.path.join(data_dir, filename)
    if os.path.exists(filepath):
        print(f"Skipping {filepath} (already exists)")
        return True

    url = f"{FINEWEB_BASE_URL}/{filename}"
    temp_path = filepath + ".tmp"
    print(f"Downloading {filename}...")
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with open(temp_path, "wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.rename(temp_path, filepath)
            print(f"Successfully downloaded {filename}")
            return True
        except (requests.RequestException, OSError) as exc:
            print(f"Attempt {attempt}/{max_attempts} failed for {filename}: {exc}")
            for path in (temp_path, filepath):
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


def download_fineweb(num_shards=FINEWEB_DEFAULT_SHARDS, num_workers=4, data_dir=None):
    """Download the first ``num_shards`` FineWeb-Edu shards into ``fineweb_dir``."""

    if num_shards < 2:
        raise ValueError(f"num_shards must be at least 2, got {num_shards}")
    data_dir = fineweb_dir(data_dir)
    os.makedirs(data_dir, exist_ok=True)

    num = min(num_shards, FINEWEB_MAX_SHARD + 1)
    ids_to_download = list(range(num))
    print(
        f"Downloading {len(ids_to_download)} shards into {data_dir} "
        f"using {num_workers} workers..."
    )
    with Pool(processes=num_workers) as pool:
        results = pool.map(
            _download_fineweb_shard, [(i, data_dir) for i in ids_to_download]
        )

    successful = sum(1 for ok in results if ok)
    print(f"Done! Downloaded {successful}/{len(ids_to_download)} shards to {data_dir}")
    return data_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download FineWeb-Edu 100BT parquet shards for ESM training"
    )
    parser.add_argument(
        "-n",
        "--num-shards",
        type=int,
        default=FINEWEB_DEFAULT_SHARDS,
        help=(
            "number of leading shards to download "
            f"(default: {FINEWEB_DEFAULT_SHARDS}); the last one is the validation split"
        ),
    )
    parser.add_argument("-w", "--num-workers", type=int, default=4)
    parser.add_argument("--data-dir", default=None, help="override the output directory")
    cli_args = parser.parse_args()
    download_fineweb(cli_args.num_shards, cli_args.num_workers, cli_args.data_dir)
