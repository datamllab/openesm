from esm.dataloader import StatefulBestFitDataLoader
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset as _IterableDataset


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
