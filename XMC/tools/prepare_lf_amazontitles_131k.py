#!/usr/bin/env python3
"""Convert FroPa LF-AmazonTitles-131K JSONL into the text format used by MHE-XMC."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED = {
    "trn": 294_805,
    "tst": 134_835,
    "labels": 131_073,
}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").replace("\t", " ").splitlines()).strip()


def convert_split(source: Path, text_output: Path, label_output: Path) -> tuple[int, set[int]]:
    seen_labels: set[int] = set()
    count = 0
    text_tmp = text_output.with_suffix(text_output.suffix + ".tmp")
    label_tmp = label_output.with_suffix(label_output.suffix + ".tmp")

    with source.open("r", encoding="utf-8") as src, text_tmp.open(
        "w", encoding="utf-8", newline="\n"
    ) as texts, label_tmp.open("w", encoding="utf-8", newline="\n") as labels:
        for count, line in enumerate(src, start=1):
            record = json.loads(line)
            target_ind = [int(label) for label in record["target_ind"]]
            if not target_ind:
                raise ValueError(f"{source}:{count} has no target labels")
            if any(label < 0 or label >= EXPECTED["labels"] for label in target_ind):
                raise ValueError(f"{source}:{count} contains an out-of-range label")
            seen_labels.update(target_ind)
            texts.write(clean_text(record.get("title")) + "\n")
            labels.write(" ".join(map(str, target_ind)) + "\n")

    text_tmp.replace(text_output)
    label_tmp.replace(label_output)
    return count, seen_labels


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="FroPa LF-AmazonTitles-131K directory")
    parser.add_argument("output", type=Path, help="Output staging directory")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "train_raw_texts.txt": None,
        "train_labels.txt": None,
        "test_raw_texts.txt": None,
        "test_labels.txt": None,
    }

    train_count, train_labels = convert_split(
        args.source / "trn.json",
        args.output / "train_raw_texts.txt",
        args.output / "train_labels.txt",
    )
    test_count, test_labels = convert_split(
        args.source / "tst.json",
        args.output / "test_raw_texts.txt",
        args.output / "test_labels.txt",
    )
    if train_count != EXPECTED["trn"] or test_count != EXPECTED["tst"]:
        raise ValueError(f"unexpected split sizes: train={train_count}, test={test_count}")

    observed = train_labels | test_labels
    if observed != set(range(EXPECTED["labels"])):
        missing = sorted(set(range(EXPECTED["labels"])) - observed)
        raise ValueError(f"expected all labels to be observed; missing first labels: {missing[:20]}")

    for name in outputs:
        path = args.output / name
        outputs[name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}

    manifest = {
        "dataset": "LF-AmazonTitles-131K",
        "source_format": "FroPa JSONL",
        "text_field": "title",
        "train_examples": train_count,
        "test_examples": test_count,
        "num_labels": EXPECTED["labels"],
        "label_encoding": "space-separated zero-based integer label IDs",
        "files": outputs,
    }
    (args.output / "mhe_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
