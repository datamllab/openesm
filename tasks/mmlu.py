"""
The MMLU dataset.
https://huggingface.co/datasets/cais/mmlu
"""

import os
from datasets import load_dataset
from tasks.common import Task, render_mc


class MMLU(Task):
    letters = ("A", "B", "C", "D")
    groups = (
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    )

    def __init__(self, subset, split, **kwargs):
        super().__init__(**kwargs)
        assert subset in ["all", "auxiliary_train"], (
            f"subset {subset} must be all|auxiliary_train"
        )
        assert split in ["train", "validation", "dev", "test"], (
            f"split {split} must be train|validation|dev|test"
        )
        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train must be split into train"
        self.subset = subset
        self.split = split

        cache_dir = None
        download_mode = None
        if os.environ.get("ESM_SFT_DATA_DIR"):
            cache_dir = os.path.join(os.environ["ESM_SFT_DATA_DIR"], "mmlu")
            if os.path.exists(cache_dir):
                download_mode = "reuse_cache_if_exists"
            else:
                cache_dir = None

        self.ds = load_dataset(
            "cais/mmlu",
            subset,
            split=split,
            cache_dir=cache_dir,
            download_mode=download_mode,
        ).shuffle(seed=42)
        if subset == "auxiliary_train":
            self.ds = self.ds.map(lambda row: row["train"], remove_columns=["train"])

    @property
    def eval_type(self):
        return "categorical"

    def num_examples(self):
        return len(self.ds)

    def get_example(self, index):
        row = self.ds[index]
        question = row["question"]
        choices = row["choices"]
        answer = row["answer"]
        subject = row["subject"]
        assert len(choices) == 4, "MMLU should have 4 choices"

        user_message = render_mc(question, self.letters, choices)
        assistant_message = self.letters[answer]
        messages = [
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
        conversation = {
            "messages": messages,
            "subject": subject,
            "letters": self.letters,
        }
        return conversation

    def evaluate(self, conversation, assistant_response):

        assert assistant_response in self.letters, (
            f"MMLU answer {assistant_response} is expected to be one of {self.letters}"
        )
        assistant_message = conversation["messages"][-1]["content"]
        return assistant_response == assistant_message
