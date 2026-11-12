"""Prepare deterministic train/validation/test splits for the SLM dataset."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from datasets import Dataset, DatasetDict

SPLIT_SEED = 42


def read_jsonl(path: Path) -> list[dict]:
    """Read non-empty JSON Lines records, including files with a UTF-8 BOM."""
    with path.open("r", encoding="utf-8-sig") as source:
        return [json.loads(line) for line in source if line.strip()]


def split_records(records: list[dict]) -> dict[str, list[dict]]:
    """Shuffle reproducibly and return 80/10/10 splits."""
    shuffled = records.copy()
    random.Random(SPLIT_SEED).shuffle(shuffled)

    train_end = int(len(shuffled) * 0.8)
    validation_end = int(len(shuffled) * 0.9)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:validation_end],
        "test": shuffled[validation_end:],
    }


def write_splits(splits: dict[str, list[dict]], output_dir: Path) -> None:
    """Write each split as JSONL and Parquet."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, rows in splits.items():
        jsonl_path = output_dir / f"{split_name}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as destination:
            for row in rows:
                destination.write(json.dumps(row, ensure_ascii=False) + "\n")

        Dataset.from_list(rows).to_parquet(str(output_dir / f"{split_name}.parquet"))
        print(f"{split_name}: {len(rows):,} examples")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split the movie-recommendation SFT JSONL dataset."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/netflix_sft_all.jsonl"),
        help="Path to the source JSONL file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Directory for generated JSONL and Parquet splits.",
    )
    args = parser.parse_args()

    records = read_jsonl(args.input)
    splits = split_records(records)
    write_splits(splits, args.output_dir)

    dataset = DatasetDict(
        {name: Dataset.from_list(rows) for name, rows in splits.items()}
    )
    print(dataset)


if __name__ == "__main__":
    main()
