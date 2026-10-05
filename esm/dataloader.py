"""
Distributed dataloaders for pretraining.

BOS-aligned bestfit:
   - Every row starts with BOS token
   - Documents packed using best-fit algorithm to minimize cropping
   - When no document fits remaining space, crops a document to fill exactly
   - 100% utilization (no padding), ~35% tokens cropped at T=2048

Compared to the original tokenizing_distributed_data_loader:
BOS-aligned loses ~35% of tokens to cropping, but ensures that
there are fewer "confusing" tokens in the train/val batches as every token can
now attend back to the BOS token and sees the full context of the document.

Fallback to the original if you have very limited data AND long documents:
the upstream dataloader implementation.
"""

import torch
import pyarrow.parquet as pq

from esm.common import get_dist_info
from esm.pretrain_dataset import (
    get_dataset_dir,
    get_owt_split_ranges,
    list_parquet_files,
    normalize_base_train_dataset,
)


EXACT_RESUME_STATE_VERSION = 1


class StatefulBestFitDataLoader:
    """
    Stateful Best-Fit Packing Dataloader with exact resume support.

    Wraps the document iteration + best-fit packing logic into a class so that
    all internal state (parquet position, doc_batch_index within a row group,
    and the in-memory doc_buffer) can be serialised and restored for bit-exact
    training resumption.
    """

    def __init__(
        self,
        tokenizer,
        B,
        T,
        split,
        device="cuda",
        resume_state_dict=None,
        buffer_size=1000,
        tokenizer_threads=4,
        tokenizer_batch_size=128,
        dataset_name="dclm",
        data_dir=None,
    ):
        self.tokenizer = tokenizer
        self.B = B
        self.T = T
        self.split = split
        self.device = device
        self.buffer_size = buffer_size
        self.tokenizer_threads = tokenizer_threads
        self.tokenizer_batch_size = tokenizer_batch_size
        self.dataset_name = normalize_base_train_dataset(dataset_name)
        self.data_dir = get_dataset_dir(
            dataset_name=self.dataset_name, data_dir=data_dir
        )

        self.bos_token = tokenizer.get_bos_token_id()
        self.row_capacity = T + 1

        self._ddp, self._ddp_rank, self._ddp_local_rank, self._ddp_world_size = (
            get_dist_info()
        )

        all_paths = list_parquet_files(
            data_dir=self.data_dir, dataset_name=self.dataset_name
        )
        assert len(all_paths) != 0, (
            f"No parquet files found for base_train_dataset={self.dataset_name!r} "
            f"in {self.data_dir}. Run `bash runs/prepare_data.sh` "
            "on a networked cpu-worker first."
        )
        if self.dataset_name == "owt":
            owt_ranges = get_owt_split_ranges(all_paths)
            self._parquet_specs = owt_ranges[split]
            self._parquet_paths = [path for path, _, _ in self._parquet_specs]
            self._total_docs = owt_ranges["total_docs"]
        else:
            self._parquet_paths = all_paths[:-1] if split == "train" else all_paths[-1:]
            self._parquet_specs = [(path, 0, None) for path in self._parquet_paths]
            self._total_docs = None

        print(
            f"[DataLoader] Dataset='{self.dataset_name}' dir='{self.data_dir}' "
            f"Split='{split}': Using {len(self._parquet_paths)} parquet file(s)"
        )
        if self.dataset_name == "owt":
            split_docs = sum(end - start for _, start, end in self._parquet_specs)
            print(
                f"[DataLoader] OWT exact document split: {split_docs} docs "
                f"(total={self._total_docs}, held_out=100000)"
            )
        if split == "train":
            print(
                f"[DataLoader] Training on files: {self._parquet_paths[0]} to {self._parquet_paths[-1]}"
            )
        else:
            print(f"[DataLoader] Validation on file: {self._parquet_paths[0]}")

        self.next_pq_idx = 0
        self.next_rg_idx = self._ddp_rank
        self.next_epoch = 1
        self.next_doc_batch_index = 0
        self.doc_buffer = []

        self._resume_state = resume_state_dict
        self._first_pass = True

        if resume_state_dict is not None:
            self._apply_resume_state(resume_state_dict)

    def state_dict(self):
        """Export the full state needed for exact resume."""
        return {
            "state_version": EXACT_RESUME_STATE_VERSION,
            "pq_idx": self.next_pq_idx,
            "rg_idx": self.next_rg_idx,
            "epoch": self.next_epoch,
            "doc_batch_index": self.next_doc_batch_index,
            "doc_buffer": [list(doc) for doc in self.doc_buffer],
            "dataset_name": self.dataset_name,
        }

    def lightweight_state_dict(self):
        """
        Export streaming position without copying doc_buffer.

        This is not sufficient for exact resume: it restores only the
        streaming cursor and drops any prefetched, unconsumed doc_buffer.
        """
        return {
            "state_version": EXACT_RESUME_STATE_VERSION,
            "pq_idx": self.next_pq_idx,
            "rg_idx": self.next_rg_idx,
            "epoch": self.next_epoch,
            "doc_batch_index": self.next_doc_batch_index,
            "dataset_name": self.dataset_name,
        }

    def _apply_resume_state(self, state):
        """Restore internal state from a checkpoint dict."""
        if (
            state.get("state_version") == EXACT_RESUME_STATE_VERSION
            and "doc_buffer" in state
        ):
            self.next_pq_idx = state["pq_idx"]
            self.next_rg_idx = state["rg_idx"]
            self.next_epoch = state["epoch"]
            self.next_doc_batch_index = state.get("doc_batch_index", 0)
            raw_buffer = state["doc_buffer"]
            self.doc_buffer = [list(doc) for doc in raw_buffer]
            print(
                f"[Exact Resume] rank={self._ddp_rank}: pq_idx={self.next_pq_idx}, "
                f"rg_idx={self.next_rg_idx}, epoch={self.next_epoch}, "
                f"doc_batch_index={self.next_doc_batch_index}, "
                f"doc_buffer_len={len(self.doc_buffer)}"
            )
        elif state.get("state_version") == EXACT_RESUME_STATE_VERSION:
            self.next_pq_idx = state["pq_idx"]
            self.next_rg_idx = state["rg_idx"]
            self.next_epoch = state["epoch"]
            self.next_doc_batch_index = state.get("doc_batch_index", 0)
            self.doc_buffer = []
            print(
                f"[Cursor-only Resume] rank={self._ddp_rank}: pq_idx={self.next_pq_idx}, "
                f"rg_idx={self.next_rg_idx}, epoch={self.next_epoch}, "
                f"doc_batch_index={self.next_doc_batch_index}, "
                f"doc_buffer_len=0"
            )
        else:
            raise ValueError(
                "Unsupported ESM dataloader state: expected "
                f"state_version={EXACT_RESUME_STATE_VERSION!r}."
            )

    def _open_row_group(self, pq_idx, rg_idx):
        """Read a single row group and return the text list."""
        filepath = self._parquet_paths[pq_idx]
        pf = pq.ParquetFile(filepath)
        if rg_idx >= pf.num_row_groups:
            return None
        rg = pf.read_row_group(rg_idx)
        return rg.column("text").to_pylist()

    def _rank_row_group_start(self, num_row_groups):
        """Return this rank's first row group, wrapping when shards are tiny."""
        return self._ddp_rank % num_row_groups

    def _row_group_stride(self, num_row_groups):
        """Stride over row groups without leaving high ranks data-starved."""
        return min(self._ddp_world_size, num_row_groups)

    def _doc_batch_iter(self):
        """
        Infinite iterator yielding (text_batch, pq_idx, rg_idx, epoch).

        On exact resume: starts at saved pq_idx/rg_idx and skips
        doc_batch_index sub-batches within the first row group.

        The first row group is taken from the versioned resume cursor.
        """
        pq_idx = self.next_pq_idx
        epoch = self.next_epoch
        first_pass = True

        while True:
            pq_idx = self.next_pq_idx if first_pass else 0
            while pq_idx < len(self._parquet_paths):
                filepath, file_start, file_end = self._parquet_specs[pq_idx]
                pf = pq.ParquetFile(filepath)
                if pf.num_row_groups == 0:
                    pq_idx += 1
                    continue
                rg_stride = self._row_group_stride(pf.num_row_groups)

                if first_pass and pq_idx == self.next_pq_idx:
                    rg_idx = self.next_rg_idx
                    if rg_idx >= pf.num_row_groups:
                        rg_idx = self._rank_row_group_start(pf.num_row_groups)
                else:
                    rg_idx = self._rank_row_group_start(pf.num_row_groups)

                skip_doc_batches = (
                    self.next_doc_batch_index
                    if (
                        first_pass
                        and pq_idx == self.next_pq_idx
                        and rg_idx == self.next_rg_idx
                    )
                    else 0
                )

                row_group_offsets = [0]
                for metadata_idx in range(pf.num_row_groups):
                    row_group_offsets.append(
                        row_group_offsets[-1]
                        + pf.metadata.row_group(metadata_idx).num_rows
                    )
                while rg_idx < pf.num_row_groups:
                    rg = pf.read_row_group(rg_idx)
                    batch = rg.column("text").to_pylist()
                    rg_start = row_group_offsets[rg_idx]
                    rg_end = row_group_offsets[rg_idx + 1]

                    selected_start = max(rg_start, file_start)
                    selected_end = min(
                        rg_end, file_end if file_end is not None else rg_end
                    )
                    if selected_end <= selected_start:
                        rg_idx += rg_stride
                        continue
                    batch = batch[selected_start - rg_start : selected_end - rg_start]
                    doc_batch_index = 0
                    for i in range(0, len(batch), self.tokenizer_batch_size):
                        if skip_doc_batches > 0:
                            skip_doc_batches -= 1
                            doc_batch_index += 1
                            continue
                        text_sub = batch[i : i + self.tokenizer_batch_size]
                        doc_batch_index += 1

                        self.next_pq_idx = pq_idx
                        self.next_rg_idx = rg_idx
                        self.next_epoch = epoch
                        self.next_doc_batch_index = doc_batch_index
                        yield text_sub, pq_idx, rg_idx, epoch
                    rg_idx += rg_stride
                pq_idx += 1
            first_pass = False
            epoch += 1

    def __iter__(self):
        """Yields (inputs, targets, state_dict) indefinitely."""
        assert self.split in ["train", "val"], "split must be 'train' or 'val'"

        B, T = self.B, self.T
        row_capacity = self.row_capacity
        doc_buffer = self.doc_buffer
        buffer_size = self.buffer_size

        doc_iter = self._doc_batch_iter()

        def refill_buffer():
            text_batch, _, _, _ = next(doc_iter)
            token_lists = self.tokenizer.encode(
                text_batch, prepend=self.bos_token, num_threads=self.tokenizer_threads
            )
            for tokens in token_lists:
                doc_buffer.append(tokens)

        use_cuda = self.device == "cuda"
        row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
        cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
        gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=self.device)
        cpu_inputs = cpu_buffer[: B * T].view(B, T)
        cpu_targets = cpu_buffer[B * T :].view(B, T)
        inputs = gpu_buffer[: B * T].view(B, T)
        targets = gpu_buffer[B * T :].view(B, T)

        while True:
            for row_idx in range(B):
                pos = 0
                while pos < row_capacity:
                    while len(doc_buffer) < buffer_size:
                        refill_buffer()

                    remaining = row_capacity - pos

                    best_idx = -1
                    best_len = 0
                    for i, doc in enumerate(doc_buffer):
                        doc_len = len(doc)
                        if doc_len <= remaining and doc_len > best_len:
                            best_idx = i
                            best_len = doc_len

                    if best_idx >= 0:
                        doc = doc_buffer.pop(best_idx)
                        doc_len = len(doc)
                        row_buffer[row_idx, pos : pos + doc_len] = torch.tensor(
                            doc, dtype=torch.long
                        )
                        pos += doc_len
                    else:
                        shortest_idx = min(
                            range(len(doc_buffer)), key=lambda i: len(doc_buffer[i])
                        )
                        doc = doc_buffer.pop(shortest_idx)
                        row_buffer[row_idx, pos : pos + remaining] = torch.tensor(
                            doc[:remaining], dtype=torch.long
                        )
                        pos += remaining

            cpu_inputs.copy_(row_buffer[:, :-1])
            cpu_targets.copy_(row_buffer[:, 1:])

            sd = self.lightweight_state_dict()

            gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
            yield inputs, targets, sd


def tokenizing_distributed_data_loader_with_state_bos_bestfit(
    tokenizer,
    B,
    T,
    split,
    tokenizer_threads=4,
    tokenizer_batch_size=128,
    device="cuda",
    resume_state_dict=None,
    dataset_name="dclm",
    data_dir=None,
    buffer_size=1000,
):
    """
    BOS-aligned dataloader with Best-Fit Cropping.

    Delegates to StatefulBestFitDataLoader.
    """
    loader = StatefulBestFitDataLoader(
        tokenizer=tokenizer,
        B=B,
        T=T,
        split=split,
        device=device,
        resume_state_dict=resume_state_dict,
        buffer_size=buffer_size,
        tokenizer_threads=tokenizer_threads,
        tokenizer_batch_size=tokenizer_batch_size,
        dataset_name=dataset_name,
        data_dir=data_dir,
    )
    yield from loader


def tokenizing_distributed_data_loader_bos_bestfit(*args, **kwargs):
    """Helper that omits state_dict from yields."""
    for (
        inputs,
        targets,
        state_dict,
    ) in tokenizing_distributed_data_loader_with_state_bos_bestfit(*args, **kwargs):
        yield inputs, targets
