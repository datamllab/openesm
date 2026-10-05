"""CORE/question-answer evaluation entry point.

The implementation is shared with :mod:`scripts.eval`; this command only
selects the CORE task so users do not need to remember the internal selector.
"""

from __future__ import annotations

import sys

from scripts.eval import main as _eval_main


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--task" not in args and not any(item.startswith("--task=") for item in args):
        args[0:0] = ["--task", "core"]
    _eval_main(args)


if __name__ == "__main__":
    main()
