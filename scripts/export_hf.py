"""Export a Lightning ESM checkpoint as a Transformers model repository."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from esm.modeling_esm import ESMForMaskedLM, ESMTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--tokenizer-dir", type=Path)
    parser.add_argument(
        "--modeling-file",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "esm" / "modeling_esm.py",
    )
    parser.add_argument(
        "--configuration-file",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "esm"
        / "configuration_esm.py",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    tokenizer_dir = args.tokenizer_dir
    if tokenizer_dir is None:
        tokenizer_dir = checkpoint.parent / "tokenizer"
    tokenizer_dir = tokenizer_dir.expanduser().resolve()

    for path in (checkpoint, tokenizer_dir / "tokenizer.pkl"):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in (args.modeling_file, args.configuration_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    output_dir.mkdir(parents=True, exist_ok=True)
    model, _ = ESMForMaskedLM.from_legacy_checkpoint(
        checkpoint, tokenizer_path=tokenizer_dir
    )
    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer = ESMTokenizer(tokenizer_file=str(tokenizer_dir / "tokenizer.pkl"))
    tokenizer.save_pretrained(output_dir)
    tokenizer_config_path = output_dir / "tokenizer_config.json"
    tokenizer_config = {}
    if tokenizer_config_path.is_file():
        tokenizer_config = json.loads(tokenizer_config_path.read_text())
    tokenizer_config.update(
        {
            "tokenizer_class": "ESMTokenizer",
            "auto_map": {"AutoTokenizer": ["modeling_esm.ESMTokenizer", None]},
        }
    )
    tokenizer_config.pop("tokenizer_file", None)
    tokenizer_config_path.write_text(
        json.dumps(tokenizer_config, indent=2, sort_keys=True) + "\n"
    )

    for filename in ("modeling_esm.py", "configuration_esm.py"):
        source = (
            args.modeling_file if filename == "modeling_esm.py" else args.configuration_file
        ).expanduser().resolve()
        shutil.copy2(source, output_dir / filename)

    token_bytes = tokenizer_dir / "token_bytes.pt"
    if token_bytes.is_file():
        shutil.copy2(token_bytes, output_dir / token_bytes.name)

    print(f"Exported Transformers model to {output_dir}")


if __name__ == "__main__":
    main()
