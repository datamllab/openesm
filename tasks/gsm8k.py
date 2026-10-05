"""
GSM8K evaluation.
https://huggingface.co/datasets/openai/gsm8k

Example problem instance:

Question:
Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer:
Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
#### 10

Notice that GSM8K uses tool calls inside << >> tags.
"""

import os
import re
from datasets import load_dataset
from tasks.common import Task


GSM_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(completion):
    """
    Extract the numerical answer after #### marker.
    Follows official code for normalization:
    https://github.com/openai/grade-school-math/blob/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/dataset.py#L28
    """
    match = GSM_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class GSM8K(Task):
    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["main", "socratic"], "GSM8K subset must be main|socratic"
        assert split in ["train", "test"], "GSM8K split must be train|test"

        cache_dir = None
        download_mode = None
        if os.environ.get("ESM_SFT_DATA_DIR"):
            cache_dir = os.path.join(os.environ["ESM_SFT_DATA_DIR"], "gsm8k")
            if os.path.exists(cache_dir):
                download_mode = "reuse_cache_if_exists"
            else:
                cache_dir = None

        self.ds = load_dataset(
            "openai/gsm8k",
            subset,
            split=split,
            cache_dir=cache_dir,
            download_mode=download_mode,
        ).shuffle(seed=42)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        """Get a single problem from the dataset."""
        row = self.ds[index]
        question = row["question"]
        answer = row["answer"]

        assistant_message_parts = []
        parts = re.split(r"(<<[^>]+>>)", answer)
        for part in parts:
            if part.startswith("<<") and part.endswith(">>"):
                inner = part[2:-2]

                if "=" in inner:
                    expr, result = inner.rsplit("=", 1)
                else:
                    expr, result = inner, ""

                assistant_message_parts.append({"type": "python", "text": expr})

                assistant_message_parts.append(
                    {"type": "python_output", "text": result}
                )
            else:
                assistant_message_parts.append({"type": "text", "text": part})

        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_message_parts},
        ]
        conversation = {
            "messages": messages,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):
        """
        Given (conversation, completion), return evaluation outcome (0 = wrong, 1 = correct)
        Note that:
        - the conversation has both user AND assistant message (containing the ground truth answer)
        - the assistant_response is usually the alternative assistant message achieved via sampling

        The evaluator accepts a plain string completion for this task.
        """
        assert isinstance(assistant_response, str), (
            "Assuming simple string response for now"
        )

        assistant_message = conversation["messages"][-1]
        assert assistant_message["role"] == "assistant", (
            "Last message must be from the Assistant"
        )
        assert isinstance(assistant_message["content"], list), (
            "This is expected to be a list of parts"
        )
        last_text_part = assistant_message["content"][-1]["text"]

        ref_num = extract_answer(last_text_part)
        pred_num = extract_answer(assistant_response)

        is_correct = int(pred_num == ref_num)
        return is_correct

    def reward(self, conversation, assistant_response):
        """
        Used during RL. To keep things simple, just re-use the evaluation above.
        Later this could be made more complex (e.g. format matching etc.)
        """
        is_correct = self.evaluate(conversation, assistant_response)
        is_correct_float = float(is_correct)
        return is_correct_float
