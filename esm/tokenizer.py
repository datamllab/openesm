"""ESM's RustBPE training and tiktoken inference tokenizer."""

import os
import copy
import pickle
from functools import lru_cache

import rustbpe
import tiktoken
from esm.common import get_base_dir

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>",
    "<|user_end|>",
    "<|assistant_start|>",
    "<|assistant_end|>",
    "<|python_start|>",
    "<|python_end|>",
    "<|output_start|>",
    "<|output_end|>",
]


SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""


class RustBPETokenizer:
    """Light wrapper around tiktoken (for efficient inference) but train with rustbpe"""

    def __init__(self, enc, bos_token, eos_token=None):
        self.enc = enc
        self.bos_token_id = self.encode_special(bos_token)

        if eos_token is not None:
            self.eos_token_id = self.encode_special(eos_token)
        else:
            self.eos_token_id = self.bos_token_id

        self.pad_token_id = self.eos_token_id

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):

        tokenizer = rustbpe.Tokenizer()

        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
        assert vocab_size_no_special >= 256, (
            f"vocab_size_no_special must be at least 256, got {vocab_size_no_special}"
        )
        tokenizer.train_from_iterator(
            text_iterator, vocab_size_no_special, pattern=SPLIT_PATTERN
        )

        pattern = tokenizer.get_pattern()
        mergeable_ranks_list = tokenizer.get_mergeable_ranks()
        mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
        tokens_offset = len(mergeable_ranks)
        special_tokens = {
            name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)
        }
        enc = tiktoken.Encoding(
            name="rustbpe",
            pat_str=pattern,
            mergeable_ranks=mergeable_ranks,
            special_tokens=special_tokens,
        )
        return cls(enc, "<|bos|>")

    @classmethod
    def from_directory(cls, tokenizer_dir):
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "rb") as f:
            enc = pickle.load(f)
        return cls(enc, "<|bos|>")

    @classmethod
    def from_pretrained(cls, tiktoken_name):

        enc = tiktoken.get_encoding(tiktoken_name)

        return cls(enc, "<|endoftext|>", "<|endoftext|>")

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def id_to_token(self, id):
        return self.enc.decode([id])

    @lru_cache(maxsize=32)
    def encode_special(self, text):
        return self.enc.encode_single_token(text)

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, append=None, num_threads=8):

        if prepend is not None:
            prepend_id = (
                prepend if isinstance(prepend, int) else self.encode_special(prepend)
            )
        if append is not None:
            append_id = (
                append if isinstance(append, int) else self.encode_special(append)
            )

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            if append is not None:
                ids.append(append_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids_row in ids:
                    ids_row.insert(0, prepend_id)
            if append is not None:
                for ids_row in ids:
                    ids_row.append(append_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")

        return ids

    def __call__(self, *args, **kwargs):
        return self.encode(*args, **kwargs)

    def decode(self, ids):
        return self.enc.decode(ids)

    def save(self, tokenizer_dir):

        os.makedirs(tokenizer_dir, exist_ok=True)
        pickle_path = os.path.join(tokenizer_dir, "tokenizer.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer encoding to {pickle_path}")

    def render_conversation(self, conversation, max_tokens=2048):
        """
        Tokenize a single Chat conversation (which we call a "doc" or "document" here).
        Returns:
        - ids: list[int] is a list of token ids of this rendered conversation
        - mask: list[int] of same length, mask = 1 for tokens that the Assistant is expected to train on.
        """

        ids, mask = [], []

        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        if conversation["messages"][0]["role"] == "system":
            conversation = copy.deepcopy(conversation)
            messages = conversation["messages"]
            assert messages[1]["role"] == "user", (
                "System message must be followed by a user message"
            )
            messages[1]["content"] = (
                messages[0]["content"] + "\n\n" + messages[1]["content"]
            )
            messages = messages[1:]
        else:
            messages = conversation["messages"]
        assert len(messages) >= 1, f"Conversation has less than 1 message: {messages}"

        bos = self.get_bos_token_id()
        user_start, user_end = (
            self.encode_special("<|user_start|>"),
            self.encode_special("<|user_end|>"),
        )
        assistant_start, assistant_end = (
            self.encode_special("<|assistant_start|>"),
            self.encode_special("<|assistant_end|>"),
        )
        python_start, python_end = (
            self.encode_special("<|python_start|>"),
            self.encode_special("<|python_end|>"),
        )
        output_start, output_end = (
            self.encode_special("<|output_start|>"),
            self.encode_special("<|output_end|>"),
        )

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            must_be_from = "user" if i % 2 == 0 else "assistant"
            assert message["role"] == must_be_from, (
                f"Message {i} is from {message['role']} but should be from {must_be_from}"
            )

            content = message["content"]

            if message["role"] == "user":
                assert isinstance(content, str), (
                    "User messages are simply expected to be strings"
                )
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message["role"] == "assistant":
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part["text"])
                        if part["type"] == "text":
                            add_tokens(value_ids, 1)
                        elif part["type"] == "python":
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part["type"] == "python_output":
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            raise ValueError(f"Unknown part type: {part['type']}")
                else:
                    raise ValueError(f"Unknown content type: {type(content)}")
                add_tokens(assistant_end, 1)

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def visualize_tokenization(self, ids, mask, with_token_id=False):
        """Small helper function useful in debugging: visualize the tokenization of render_conversation"""
        RED = "\033[91m"
        GREEN = "\033[92m"
        RESET = "\033[0m"
        GRAY = "\033[90m"
        tokens = []
        for i, (token_id, mask_val) in enumerate(zip(ids, mask)):
            token_str = self.decode([token_id])
            color = GREEN if mask_val == 1 else RED
            tokens.append(f"{color}{token_str}{RESET}")
            if with_token_id:
                tokens.append(f"{GRAY}({token_id}){RESET}")
        return "|".join(tokens)

    def render_for_completion(self, conversation):
        """
        Used during Reinforcement Learning. In that setting, we want to
        render the conversation priming the Assistant for a completion.
        Unlike the Chat SFT case, we don't need to return the mask.
        """

        conversation = copy.deepcopy(conversation)
        messages = conversation["messages"]
        assert messages[-1]["role"] == "assistant", (
            "Last message must be from the Assistant"
        )
        messages.pop()

        ids, mask = self.render_conversation(conversation)

        assistant_start = self.encode_special("<|assistant_start|>")
        ids.append(assistant_start)
        return ids


def get_tokenizer(tokenizer_dir=None):
    if tokenizer_dir is None:
        from esm.common import get_base_dir

        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    return RustBPETokenizer.from_directory(tokenizer_dir)


def get_token_bytes(device="cpu", tokenizer_dir=None):
    import torch

    if tokenizer_dir is None:
        from esm.common import get_base_dir

        tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
    token_bytes_path = os.path.join(tokenizer_dir, "token_bytes.pt")
    assert os.path.exists(token_bytes_path), (
        f"Token bytes not found at {token_bytes_path}? Prepare the tokenizer assets first."
    )
    with open(token_bytes_path, "rb") as f:
        token_bytes = torch.load(f, map_location=device)
    return token_bytes


class ESMTokenizerWrapper:
    """
    Wrapper around the ESM BPE tokenizer to provide a HuggingFace-compatible interface.
    This allows it to be used as a drop-in replacement in ESM code.
    """

    def __init__(self, tokenizer_obj=None, tokenizer_dir=None):
        """
        Initialize wrapper with either an existing tokenizer object or by loading from directory.

        Args:
            tokenizer_obj: Optional existing RustBPETokenizer instance
            tokenizer_dir: Directory to load tokenizer from if tokenizer_obj is None
        """
        if tokenizer_obj is not None:
            self.tokenizer = tokenizer_obj
        else:
            if tokenizer_dir is None:
                tokenizer_dir = os.path.join(get_base_dir(), "tokenizer")
            self.tokenizer = RustBPETokenizer.from_directory(tokenizer_dir)

        self.eos_token_id = self.tokenizer.get_bos_token_id()
        self.bos_token_id = self.tokenizer.get_bos_token_id()
        self.pad_token_id = self.eos_token_id
        self.unk_token_id = 0

        print(f"[ESMTokenizerWrapper] Vocab size: {self.tokenizer.get_vocab_size()}")
        print(f"[ESMTokenizerWrapper] EOS/BOS/PAD token ID: {self.eos_token_id}")

    def __len__(self):
        """Return vocab size"""
        return self.tokenizer.get_vocab_size()

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        **kwargs,
    ):
        """
        HuggingFace-style __call__ interface for tokenization.

        Args:
            text: str or list of str
            return_tensors: 'pt' for PyTorch tensors, None for lists
            padding: bool or str, whether to pad sequences
            truncation: bool, whether to truncate sequences
            max_length: int, maximum sequence length

        Returns:
            dict with 'input_ids' and 'attention_mask'
        """
        import torch

        if isinstance(text, str):
            ids = self.tokenizer.encode(text)
            ids_list = [ids]
        elif isinstance(text, (list, tuple)):
            ids_list = [self.tokenizer.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

        if truncation and max_length is not None:
            ids_list = [ids[:max_length] for ids in ids_list]

        if padding:
            if max_length is not None:
                target_length = max_length
            else:
                target_length = max(len(ids) for ids in ids_list)

            padded_ids = []
            attention_masks = []
            for ids in ids_list:
                seq_len = len(ids)
                if seq_len < target_length:
                    padded = ids + [self.pad_token_id] * (target_length - seq_len)
                    mask = [1] * seq_len + [0] * (target_length - seq_len)
                else:
                    padded = ids
                    mask = [1] * len(ids)
                padded_ids.append(padded)
                attention_masks.append(mask)
        else:
            padded_ids = ids_list
            attention_masks = [[1] * len(ids) for ids in ids_list]

        if return_tensors == "pt":
            import torch

            input_ids = torch.tensor(padded_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_masks, dtype=torch.long)
        else:
            input_ids = padded_ids
            attention_mask = attention_masks

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def encode(self, text, add_special_tokens=False, **kwargs):
        """Encode text to token IDs"""
        if isinstance(text, str):
            return self.tokenizer.encode(text)
        elif isinstance(text, (list, tuple)):
            return [self.tokenizer.encode(t) for t in text]
        else:
            raise ValueError(f"Unsupported text type: {type(text)}")

    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        """Decode token IDs to text"""
        import torch

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()

        if skip_special_tokens:
            special_token_ids = {
                self.bos_token_id,
                self.eos_token_id,
                self.pad_token_id,
                self.unk_token_id,
            }

            for name in [
                "<|user_start|>",
                "<|user_end|>",
                "<|assistant_start|>",
                "<|assistant_end|>",
                "<|python_start|>",
                "<|python_end|>",
                "<|output_start|>",
                "<|output_end|>",
            ]:
                tid = (
                    self.tokenizer.encode_special(name)
                    if hasattr(self.tokenizer, "encode_special")
                    else None
                )
                if tid is not None:
                    special_token_ids.add(tid)
            token_ids = [tid for tid in token_ids if tid not in special_token_ids]

        return self.tokenizer.decode(token_ids)

    def batch_decode(self, sequences, skip_special_tokens=False, **kwargs):
        """Batch decode token IDs to text"""
        import torch

        if isinstance(sequences, torch.Tensor):
            sequences = sequences.tolist()
        result = []
        for seq in sequences:
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()

            if skip_special_tokens:
                special_token_ids = {
                    self.bos_token_id,
                    self.eos_token_id,
                    self.pad_token_id,
                    self.unk_token_id,
                }
                for name in [
                    "<|user_start|>",
                    "<|user_end|>",
                    "<|assistant_start|>",
                    "<|assistant_end|>",
                    "<|python_start|>",
                    "<|python_end|>",
                    "<|output_start|>",
                    "<|output_end|>",
                ]:
                    tid = (
                        self.tokenizer.encode_special(name)
                        if hasattr(self.tokenizer, "encode_special")
                        else None
                    )
                    if tid is not None:
                        special_token_ids.add(tid)
                seq = [tid for tid in seq if tid not in special_token_ids]

            result.append(self.tokenizer.decode(seq))
        return result


def get_esm_tokenizer():
    """
    Convenience function to get the esm tokenizer with HuggingFace-compatible interface.

    Returns:
        ESMTokenizerWrapper instance
    """
    return ESMTokenizerWrapper()


__all__ = [
    "SPECIAL_TOKENS",
    "SPLIT_PATTERN",
    "RustBPETokenizer",
    "get_tokenizer",
    "get_token_bytes",
    "ESMTokenizerWrapper",
    "get_esm_tokenizer",
]
