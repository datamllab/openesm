"""Local datasets and MDLM-compatible block construction for zero-shot eval."""

from __future__ import annotations

import json
import os
import re

import numpy as np


DATA_ROOT_DEFAULT = os.environ.get("ESM_ZERO_SHOT_DATA_ROOT", "data/zeroshot")
LM1B_HELDOUT = [
    f"LM1B/heldout-monolingual.tokenized.shuffled/news.en.heldout-{i:05d}-of-00050"
    for i in range(50)
]


def wt_detokenizer(text):
    text = text.replace("s '", "s'")
    text = re.sub(r"/' [0-9]/", r"/'[0-9]/", text)
    for source, target in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", ".")):
        text = text.replace(source, target)
    for source, target in (
        (" : ", ": "),
        (" ; ", "; "),
        (" . ", ". "),
        (" ! ", "! "),
        (" ? ", "? "),
        (" , ", ", "),
    ):
        text = text.replace(source, target)
    text = re.sub(r"\(\s*([^\)]*?)\s*\)", r"(\1)", text)
    text = re.sub(r"\[\s*([^\]]*?)\s*\]", r"[\1]", text)
    text = re.sub(r"{\s*([^}]*?)\s*}", r"{\1}", text)
    text = re.sub(r"\"\s*([^\"]*?)\s*\"", r'"\1"', text)
    text = re.sub(r"'\s*([^']*?)\s*'", r"'\1'", text)
    for source, target in (
        ("= = = =", "===="),
        ("= = =", "==="),
        ("= =", "=="),
        (" \n", "\n"),
        ("\n ", "\n"),
        (" N ", " 1 "),
        (" 's", "'s"),
    ):
        text = text.replace(source, target)
    text = text.replace(" " + chr(176) + " ", chr(176))
    return text


def ptb_detokenizer(text):
    text = text.replace(" 's", "'s")
    text = text.replace("s ' ", "s' ")
    text = text.replace(" n't", "n't")
    text = text.replace(" \n ", "\n").replace("\\/", "/")
    for _ in range(10):
        text = text.replace(" N ", " 1 ")
    return text.replace("$ 1", "$1").replace("# 1", "#1").replace("<unk>", "?")


def lm1b_detokenizer(text):
    text = text.replace("http : / / ", "http://").replace("https : / / ", "https://")
    text = re.sub(r" \'(\w+)", r"'\1", text)
    text = re.sub(r" (\w+) \. ", r" \1. ", text)
    text = re.sub(r" (\w+) \.$", r" \1.", text)
    for source, target in (
        (" ? ", "? "),
        (" ! ", "! "),
        (" , ", ", "),
        (" : ", ": "),
        (" ; ", "; "),
        (" / ", "/"),
        ("$ ", "$"),
        ("£ ", "£"),
    ):
        text = text.replace(source, target)
    text = re.sub(r" \?$", "?", text)
    text = re.sub(r" \!$", "!", text)
    text = re.sub(r'" ([^\"]+) "', r'"\1"', text)
    text = re.sub(r"\' ([^\']+) \'", r"'\1'", text)
    text = re.sub(r"\( ([^\(\)]+) \)", r"(\1)", text)
    text = re.sub(r"\[ ([^\[\]]+) \]", r"[\1]", text)
    return text


def lambada_detokenizer(text):
    return "\n" + text.replace("“", '"').replace("”", '"').strip()


def scientific_papers_detokenizer(text):
    return lm1b_detokenizer(wt_detokenizer(text))


DETOKENIZERS = {
    "wt": wt_detokenizer,
    "ptb": ptb_detokenizer,
    "lm1b": lm1b_detokenizer,
    "lambada": lambada_detokenizer,
    "scipapers": scientific_papers_detokenizer,
    None: None,
}


DATASETS = {
    "ptb": {
        "kind": "parquet",
        "files": ["PTB/penn_treebank/validation/0000.parquet"],
        "field": "sentence",
        "detok": "ptb",
    },
    "wikitext103": {
        "kind": "parquet",
        "files": ["Wikitext/wikitext-103-raw-v1/validation-00000-of-00001.parquet"],
        "field": "text",
        "detok": "wt",
    },
    "lm1b": {"kind": "lines", "files": LM1B_HELDOUT, "detok": "lm1b"},
    "lambada": {
        "kind": "jsonl",
        "files": ["Lambada/lambada_test.jsonl"],
        "field": "text",
        "detok": "lambada",
    },
    "ag_news": {
        "kind": "parquet",
        "files": ["AG_News/data/test-00000-of-00001.parquet"],
        "field": "text",
        "detok": None,
    },
    "pubmed": {"kind": "scipapers", "files": ["Pubmed/val.txt"], "detok": "scipapers"},
    "arxiv": {"kind": "scipapers", "files": ["Arxiv/val.txt"], "detok": "scipapers"},
}
EVAL_ORDER = ["ptb", "wikitext103", "lm1b", "lambada", "ag_news", "pubmed", "arxiv"]


def _paths(name, data_root):
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; choose from {EVAL_ORDER}")
    paths = [os.path.join(data_root, rel) for rel in DATASETS[name]["files"]]
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"{name}: missing local files: {missing}")
    return paths


def _read_parquet(paths, field):
    import pyarrow.parquet as pq

    values = []
    for path in paths:
        table = pq.read_table(path, columns=[field])
        values.extend(table.column(field).to_pylist())
    return values


def _read_lines(paths):
    values = []
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as handle:
            values.extend(line.strip() for line in handle if line.strip())
    return values


def _read_jsonl(paths, field):
    values = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append(json.loads(line)[field])
    return values


def _read_scientific_papers(paths):
    values = []
    for path in paths:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    values.append("\n".join(json.loads(line)["article_text"]))
    return values


def load_texts(name, data_root=DATA_ROOT_DEFAULT):
    spec = DATASETS[name]
    paths = _paths(name, data_root)
    if spec["kind"] == "parquet":
        texts = _read_parquet(paths, spec["field"])
    elif spec["kind"] == "lines":
        texts = _read_lines(paths)
    elif spec["kind"] == "jsonl":
        texts = _read_jsonl(paths, spec["field"])
    else:
        texts = _read_scientific_papers(paths)
    if not texts:
        raise RuntimeError(f"{name}: loaded zero documents")
    return texts


def build_blocks(name, tokenizer, data_root, block_size, cache_dir, log=print):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir, f"zeroshot_blocks_{name}_bs{block_size}_v{tokenizer.get_vocab_size()}.npy"
    )
    if os.path.isfile(cache_path):
        blocks = np.load(cache_path)
        log(f"[{name}] loaded cached blocks {blocks.shape}")
        return blocks

    texts = load_texts(name, data_root)
    detokenizer = DETOKENIZERS[DATASETS[name]["detok"]]
    if detokenizer is not None:
        texts = [detokenizer(text) for text in texts]
    bos = tokenizer.get_bos_token_id()
    flat = []
    chunk_size = 2000
    for offset in range(0, len(texts), chunk_size):
        token_rows = tokenizer.encode(
            texts[offset : offset + chunk_size], append=bos, num_threads=16
        )
        for row in token_rows:
            flat.extend(row)
    inner = block_size - 2
    block_count = len(flat) // inner
    if block_count == 0:
        raise RuntimeError(f"{name}: not enough tokens for one block")
    content = np.asarray(flat[: block_count * inner], dtype=np.int64).reshape(block_count, inner)
    blocks = np.empty((block_count, block_size), dtype=np.int64)
    blocks[:, 0] = bos
    blocks[:, 1:-1] = content
    blocks[:, -1] = bos
    np.save(cache_path, blocks)
    log(f"[{name}] {len(texts)} documents -> {blocks.shape}; cached at {cache_path}")
    return blocks


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default=DATA_ROOT_DEFAULT)
    parser.add_argument("--dataset", choices=EVAL_ORDER)
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()
    if args.list or args.dataset is None:
        for name in EVAL_ORDER:
            print(name, DATASETS[name]["kind"], DATASETS[name]["files"][:1])
        return
    print(f"{args.dataset}: {len(load_texts(args.dataset, args.data_root))} documents")


if __name__ == "__main__":
    main()
