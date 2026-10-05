"""
SmolTalk by HuggingFace. Good "general" conversational dataset.
https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk
We use the "smol" version, which is more appropriate for smaller models.
"""

import os
from datasets import load_dataset
from tasks.common import Task


class SmolTalk(Task):
    """smol-smoltalk dataset. train is 460K rows, test is 24K rows."""

    def __init__(self, split, **kwargs):
        super().__init__(**kwargs)
        assert split in ["train", "test"], "SmolTalk split must be train|test"

        cache_dir = None
        download_mode = None
        if os.environ.get("ESM_SFT_DATA_DIR"):
            cache_dir = os.path.join(os.environ["ESM_SFT_DATA_DIR"], "smol-smoltalk")
            if os.path.exists(cache_dir):
                download_mode = "reuse_cache_if_exists"
            else:
                cache_dir = None

        self.ds = load_dataset(
            "HuggingFaceTB/smol-smoltalk",
            split=split,
            cache_dir=cache_dir,
            download_mode=download_mode,
        ).shuffle(seed=42)
        self.length = len(self.ds)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        row = self.ds[index]
        messages = row["messages"]

        assert len(messages) >= 1
        first_message = messages[0]
        if first_message["role"] == "system":
            rest_messages = messages[1:]
        else:
            rest_messages = messages
        assert len(rest_messages) >= 2, (
            "SmolTalk messages must have at least 2 messages"
        )
        for i, message in enumerate(rest_messages):
            expected_role = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == expected_role, (
                f"Message {i} has role {message['role']} but should be {expected_role}"
            )
            assert isinstance(message["content"], str), "Content must be a string"

        conversation = {
            "messages": messages,
        }
        return conversation
