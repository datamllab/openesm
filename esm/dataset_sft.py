"""
ESM SFT dataset adapter for ESM task mixtures.

This keeps the packed `(inputs, targets)` interface expected by the ESM trainer
while using ESM conversation tasks as the data source.
"""

import copy
import fcntl
import hashlib
import logging
import os
import time
from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as _IterableDataset

from esm.common import get_dist_info


def _sft_debug(message):
    if os.environ.get("ESM_SFT_DEBUG", "0").lower() not in ("1", "true", "yes"):
        return
    rank = os.environ.get("RANK", "?")
    debug_ranks = os.environ.get("ESM_SFT_DEBUG_RANKS", "0").strip().lower()
    if debug_ranks not in ("all", "*"):
        enabled_ranks = {
            item.strip() for item in debug_ranks.split(",") if item.strip()
        }
        if rank not in enabled_ranks:
            return
    local_rank = os.environ.get("LOCAL_RANK", "?")
    print(f"[ESM SFT][rank={rank} local={local_rank}] {message}", flush=True)


def _build_logged_task(name, factory):
    start = time.monotonic()
    _sft_debug(f"task_start name={name}")
    try:
        task = factory()
        _sft_debug(
            f"task_done name={name} len={len(task)} elapsed={time.monotonic() - start:.2f}s"
        )
        return task
    except Exception as exc:
        _sft_debug(
            f"task_error name={name} elapsed={time.monotonic() - start:.2f}s error={type(exc).__name__}: {exc}"
        )
        raise


_SFT_DATASETS_CACHE = None
_SFT_DATASET_LOGGING_CONFIGURED = False


def _build_sft_inputs_and_targets(batch_tensor, mask_tensor, device, use_cuda):
    inputs = batch_tensor[:, :-1].to(
        device=device, dtype=torch.int64, non_blocking=use_cuda
    )
    targets = batch_tensor[:, 1:].to(
        device=device, dtype=torch.int64, non_blocking=use_cuda
    )
    target_mask = mask_tensor[:, 1:].to(
        device=device, dtype=torch.bool, non_blocking=use_cuda
    )

    targets = targets.masked_fill(~target_mask, -1)
    return inputs, targets


@contextmanager
def _sft_dataset_load_lock():
    """Serialize HF cached dataset construction across local DDP ranks.

    SFT currently runs on a single node, so the default /tmp lock is
    intentionally node-local. If SFT is expanded to multi-node with a shared
    writable dataset cache, set ESM_SFT_DATASET_LOAD_LOCK_PATH to a shared
    filesystem path.
    """
    if os.environ.get("ESM_SFT_DATASET_LOAD_LOCK", "1").lower() in ("0", "false", "no"):
        yield
        return

    sft_data_dir = os.environ.get("ESM_SFT_DATA_DIR", "")
    lock_key = hashlib.sha1(sft_data_dir.encode("utf-8")).hexdigest()[:12]
    lock_path = os.environ.get(
        "ESM_SFT_DATASET_LOAD_LOCK_PATH",
        os.path.join("/tmp", f"esm_sft_dataset_load_{lock_key}.lock"),
    )
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    with open(lock_path, "w") as lock_file:
        wait_start = time.monotonic()
        _sft_debug(f"load_lock_wait path={lock_path}")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        _sft_debug(
            f"load_lock_acquired path={lock_path} wait={time.monotonic() - wait_start:.2f}s"
        )
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            _sft_debug(f"load_lock_release path={lock_path}")


def _silence_hf_dataset_warnings():
    """Keep multi-rank cached dataset probing from flooding the shared stdout pipe."""
    global _SFT_DATASET_LOGGING_CONFIGURED
    if _SFT_DATASET_LOGGING_CONFIGURED:
        return
    for name in (
        "datasets",
        "datasets.load",
        "datasets.builder",
        "datasets.packaged_modules.cache.cache",
    ):
        logging.getLogger(name).setLevel(logging.ERROR)
    try:
        from datasets.utils.logging import set_verbosity_error

        set_verbosity_error()
    except Exception:
        pass
    _SFT_DATASET_LOGGING_CONFIGURED = True


def _load_sft_datasets(caller_split=None):
    """Load the ESM SFT train/validation task mixtures."""
    global _SFT_DATASETS_CACHE

    if _SFT_DATASETS_CACHE is not None:
        train_dataset, val_dataset = _SFT_DATASETS_CACHE
        _sft_debug(
            f"load_cache_hit caller_split={caller_split} train_len={len(train_dataset)} "
            f"val_len={len(val_dataset)}"
        )
        return _SFT_DATASETS_CACHE

    _silence_hf_dataset_warnings()

    from tasks.common import TaskMixture
    from tasks.gsm8k import GSM8K
    from tasks.mmlu import MMLU
    from tasks.smoltalk import SmolTalk

    with _sft_dataset_load_lock():
        load_start = time.monotonic()
        sft_data_dir = os.environ.get("ESM_SFT_DATA_DIR", "")
        _sft_debug(
            f"load_start caller_split={caller_split} sft_data_dir={sft_data_dir}"
        )
        if sft_data_dir and os.path.exists(sft_data_dir):
            _sft_debug(f"using_offline_dataset_root path={sft_data_dir}")

        base_dir = os.environ.get("ESM_BASE_DIR", "")
        identity_path = os.path.join(base_dir, "identity_conversations.jsonl")

        train_tasks = [
            _build_logged_task("train/smoltalk", lambda: SmolTalk(split="train")),
            _build_logged_task(
                "train/mmlu_auxiliary_train",
                lambda: MMLU(subset="auxiliary_train", split="train"),
            ),
            _build_logged_task(
                "train/gsm8k_main_1", lambda: GSM8K(subset="main", split="train")
            ),
            _build_logged_task(
                "train/gsm8k_main_2", lambda: GSM8K(subset="main", split="train")
            ),
        ]

        try:
            from tasks.customjson import CustomJSON

            if os.path.exists(identity_path):
                train_tasks.append(
                    _build_logged_task(
                        "train/customjson_identity_1",
                        lambda: CustomJSON(filepath=identity_path),
                    )
                )
                train_tasks.append(
                    _build_logged_task(
                        "train/customjson_identity_2",
                        lambda: CustomJSON(filepath=identity_path),
                    )
                )
        except Exception as exc:
            _sft_debug(
                f"optional_task_skip name=customjson error={type(exc).__name__}: {exc}"
            )
            pass

        try:
            from tasks.spellingbee import SimpleSpelling, SpellingBee

            train_tasks.append(
                _build_logged_task(
                    "train/simple_spelling",
                    lambda: SimpleSpelling(size=200000, split="train"),
                )
            )
            train_tasks.append(
                _build_logged_task(
                    "train/spellingbee", lambda: SpellingBee(size=80000, split="train")
                )
            )
        except Exception as exc:
            _sft_debug(
                f"optional_task_skip name=spelling error={type(exc).__name__}: {exc}"
            )
            pass

        train_dataset = _build_logged_task(
            "train/task_mixture", lambda: TaskMixture(train_tasks)
        )

        val_tasks = [
            _build_logged_task("val/smoltalk", lambda: SmolTalk(split="test")),
            _build_logged_task(
                "val/mmlu_all", lambda: MMLU(subset="all", split="test", stop=5200)
            ),
            _build_logged_task(
                "val/gsm8k_main", lambda: GSM8K(subset="main", split="test", stop=420)
            ),
        ]
        val_dataset = _build_logged_task(
            "val/task_mixture", lambda: TaskMixture(val_tasks)
        )

        _sft_debug(
            f"load_done caller_split={caller_split} train_len={len(train_dataset)} "
            f"val_len={len(val_dataset)} elapsed={time.monotonic() - load_start:.2f}s"
        )

    _SFT_DATASETS_CACHE = (train_dataset, val_dataset)
    return _SFT_DATASETS_CACHE


class SFTIterableDataset(_IterableDataset):
    """Adapt ESM SFT task mixtures to the ESM iterable dataloader API."""

    STATE_VERSION = "sft_v1"

    def __init__(
        self,
        tokenizer,
        batch_size,
        max_len,
        split,
        max_iter,
        device="cuda",
        resume_state_dict=None,
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.B = batch_size
        self.T = max_len
        self.split = split
        self.max_iter = max_iter
        self.device = device
        self.resume_state_dict = resume_state_dict
        self._resume_state_locked = self._can_restore(resume_state_dict)
        self.last_state_dict = None
        self._runtime_initialized = False

        self.row_capacity = self.T + 1
        self.buffer_size = 1000
        self.conv_buffer = []
        self.cursor = None
        self.consumed = None
        self.epoch = 1
        self.it = 0
        self.ddp_rank = None
        self.ddp_world_size = None
        self._debug_first_yield_logged = False

        _sft_debug(
            f"dataset_init_start split={self.split} batch_size={self.B} max_len={self.T} "
            f"max_iter={self.max_iter} device={self.device}"
        )
        self.train_dataset, self.val_dataset = _load_sft_datasets(
            caller_split=self.split
        )
        _sft_debug(
            f"dataset_init_done split={self.split} train_len={len(self.train_dataset)} "
            f"val_len={len(self.val_dataset)}"
        )

    def _can_restore(self, state):
        return (
            isinstance(state, dict) and state.get("state_version") == self.STATE_VERSION
        )

    def _initialize_fresh_state(self, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = self.T + 1
        self.buffer_size = 1000
        self.conv_buffer = []
        self.cursor = ddp_rank
        self.consumed = ddp_rank
        self.epoch = 1
        self.it = 0

    def _restore_state(self, state, ddp_rank, ddp_world_size):
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.row_capacity = int(state.get("row_capacity", self.T + 1))
        self.buffer_size = int(state.get("buffer_size", 1000))
        self.conv_buffer = copy.deepcopy(state.get("conv_buffer", []))
        self.cursor = int(state.get("cursor", ddp_rank))
        self.consumed = int(state.get("consumed", ddp_rank))
        self.epoch = int(state.get("epoch", 1))
        self.it = int(state.get("it", 0))
        print(
            f"[SFT Resume] Rank {ddp_rank} restored state: "
            f"cursor={self.cursor}, consumed={self.consumed}, epoch={self.epoch}, "
            f"it={self.it}, conv_buffer={len(self.conv_buffer)}"
        )

    def _build_state_dict(self, copy_buffer):
        return {
            "state_version": self.STATE_VERSION,
            "split": self.split,
            "cursor": self.cursor,
            "consumed": self.consumed,
            "epoch": self.epoch,
            "it": self.it,
            "buffer_size": self.buffer_size,
            "row_capacity": self.row_capacity,
            "rank": self.ddp_rank,
            "world_size": self.ddp_world_size,
            "conv_buffer": copy.deepcopy(self.conv_buffer)
            if copy_buffer
            else self.conv_buffer,
        }

    def __iter__(self):
        """BOS-aligned best-fit packing, matching ESM chat formatting."""
        ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
        dataset = self.train_dataset if self.split == "train" else self.val_dataset
        dataset_size = len(dataset)
        assert dataset_size > 0
        _sft_debug(
            f"iter_start split={self.split} ddp_rank={ddp_rank} local_rank={ddp_local_rank} "
            f"world_size={ddp_world_size} dataset_size={dataset_size}"
        )

        bos_token = self.tokenizer.get_bos_token_id()
        use_cuda = "cuda" in str(self.device)
        if self.split == "train":
            if not self._runtime_initialized:
                if self._can_restore(self.resume_state_dict):
                    self._restore_state(
                        self.resume_state_dict, ddp_rank, ddp_world_size
                    )
                else:
                    self._initialize_fresh_state(ddp_rank, ddp_world_size)
                self._runtime_initialized = True
        else:
            self._initialize_fresh_state(ddp_rank, ddp_world_size)

        _sft_debug(
            f"iter_state_ready split={self.split} bos_token={bos_token} cursor={self.cursor} "
            f"buffer={len(self.conv_buffer)}"
        )

        first_refill_logged = False

        def refill_buffer():
            nonlocal first_refill_logged
            if not first_refill_logged:
                _sft_debug(
                    f"refill_start split={self.split} cursor={self.cursor} "
                    f"buffer={len(self.conv_buffer)} target={self.buffer_size}"
                )
            while len(self.conv_buffer) < self.buffer_size:
                conversation = dataset[self.cursor]
                ids, mask = self.tokenizer.render_conversation(conversation)
                self.cursor += ddp_world_size
                if self.cursor >= dataset_size:
                    self.cursor = self.cursor % dataset_size
                    self.epoch += 1
                if len(ids) <= self.row_capacity:
                    self.conv_buffer.append((ids, mask))
            if not first_refill_logged:
                _sft_debug(
                    f"refill_done split={self.split} cursor={self.cursor} "
                    f"buffer={len(self.conv_buffer)} epoch={self.epoch}"
                )
                first_refill_logged = True

        while True:
            if self.split == "train" and self.it >= self.max_iter:
                break

            rows = []
            row_masks = []

            for _ in range(self.B):
                row = []
                row_mask = []
                while len(row) < self.row_capacity:
                    while len(self.conv_buffer) < self.buffer_size:
                        refill_buffer()
                    remaining = self.row_capacity - len(row)

                    best_idx = -1
                    best_len = 0
                    for i, (conv_ids, conv_mask) in enumerate(self.conv_buffer):
                        conv_len = len(conv_ids)
                        if conv_len <= remaining and conv_len > best_len:
                            best_idx = i
                            best_len = conv_len

                    if best_idx >= 0:
                        conv_ids, conv_mask = self.conv_buffer.pop(best_idx)
                        row.extend(conv_ids)
                        row_mask.extend(conv_mask)
                        self.consumed += ddp_world_size
                    else:
                        remaining = self.row_capacity - len(row)
                        row.extend([bos_token] * remaining)
                        row_mask.extend([0] * remaining)
                        break

                rows.append(row[: self.row_capacity])
                row_masks.append(row_mask[: self.row_capacity])

            batch_tensor = torch.tensor(rows, dtype=torch.long, pin_memory=use_cuda)
            mask_tensor = torch.tensor(row_masks, dtype=torch.bool, pin_memory=use_cuda)
            inputs, targets = _build_sft_inputs_and_targets(
                batch_tensor, mask_tensor, self.device, use_cuda
            )

            self.it += 1
            self.last_state_dict = self._build_state_dict(copy_buffer=False)
            if not self._debug_first_yield_logged:
                supervised_tokens = int((targets != -1).sum().detach().cpu().item())
                _sft_debug(
                    f"yield_first_batch split={self.split} it={self.it} "
                    f"inputs_shape={tuple(inputs.shape)} supervised_tokens={supervised_tokens} "
                    f"cursor={self.cursor} epoch={self.epoch}"
                )
                self._debug_first_yield_logged = True
            yield inputs, targets

            if self.split != "train" and self.it >= self.max_iter:
                break

    def get_dataloader_state(self):
        if self.cursor is None:
            return self.last_state_dict
        return self._build_state_dict(copy_buffer=True)

    def state_dict(self):
        state = self.get_dataloader_state()
        return {} if state is None else state

    def load_state_dict(self, state_dict):
        if self._resume_state_locked and self._can_restore(self.resume_state_dict):
            current_state = self.resume_state_dict
            incoming_state = state_dict
            if current_state != incoming_state:
                print(
                    f"[SFT Resume] Ignore external dataloader overwrite: "
                    f"current_rank={current_state.get('rank')}, "
                    f"incoming_rank={incoming_state.get('rank') if isinstance(incoming_state, dict) else None}"
                )
                return
        self.resume_state_dict = copy.deepcopy(state_dict)
        self._runtime_initialized = False

    def __len__(self):
        return self.max_iter


class SFTDataLoader(DataLoader):
    def __len__(self):
        dataset = getattr(self, "dataset", None)
        split = getattr(dataset, "split", None)
        if split == "train" and os.environ.get(
            "ESM_SFT_HIDE_TRAIN_DATALOADER_LEN", "1"
        ).lower() not in (
            "0",
            "false",
            "no",
        ):
            raise TypeError(
                "SFT train dataloader is intentionally unsized for Lightning DDP"
            )
        return super().__len__()

    def __iter__(self):
        dataset = getattr(self, "dataset", None)
        split = getattr(dataset, "split", "?")
        _sft_debug(f"dataloader_iter_enter split={split}")
        iterator = super().__iter__()
        _sft_debug(
            f"dataloader_iter_return split={split} iterator={type(iterator).__name__}"
        )
        return iterator

    def state_dict(self):
        dataset = getattr(self, "dataset", None)
        if dataset is None or not hasattr(dataset, "state_dict"):
            return {}
        return dataset.state_dict()

    def load_state_dict(self, state_dict):
        dataset = getattr(self, "dataset", None)
        if dataset is not None and hasattr(dataset, "load_state_dict"):
            dataset.load_state_dict(state_dict)


def generate_sft_dataloader(
    tokenizer, batch_size, max_len, max_iter, split, device, resume_state_dict=None
):
    dataset = SFTIterableDataset(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        split=split,
        max_iter=max_iter,
        device=device,
        resume_state_dict=resume_state_dict,
    )
    dataloader = SFTDataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )
    return dataloader
