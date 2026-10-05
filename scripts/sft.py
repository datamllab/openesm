"""Supervised fine-tuning entry point.

SFT uses the same ESM trainer/model implementation as pretraining. This
small task entry point supplies the only task-specific default and leaves all
other parameters to ``configs/train.yaml`` and explicit CLI overrides.
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    args = list(sys.argv[1:])
    if "--execution_mode" not in args and not any(
        item.startswith("--execution_mode=") for item in args
    ):
        args[0:0] = ["--execution_mode", "finetune"]
    sys.argv = ["scripts.train", *args]
    runpy.run_module("scripts.train", run_name="__main__")


if __name__ == "__main__":
    main()
