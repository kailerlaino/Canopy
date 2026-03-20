"""Load canopy_input.jsonl dataset for evolutionary mimicry."""

import json
import random
from pathlib import Path
from typing import TypedDict


class DatasetRecord(TypedDict):
    """Single record from canopy_input.jsonl."""

    state: list[float]
    action_no_noise: list[float]


def load_canopy_dataset(path: str | Path) -> list[DatasetRecord]:
    """
    Load canopy_input.jsonl and return list of {state, action_no_noise} records.

    Args:
        path: Path to canopy_input.jsonl file.

    Returns:
        List of records with 'state' (list of floats) and 'action_no_noise' (list with 1 float).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records: list[DatasetRecord] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            state = rec.get("state", [])
            action_no_noise = rec.get("action_no_noise", rec.get("action", [0.0]))
            if isinstance(action_no_noise, list):
                action_no_noise = action_no_noise[0] if action_no_noise else 0.0
            records.append(
                {
                    "state": state,
                    "action_no_noise": [float(action_no_noise)],
                }
            )

    return records


def load_canopy_dataset_dir(path: str | Path) -> list[DatasetRecord]:
    """
    Load all canopy_input.jsonl files under the directory and return combined records.

    Args:
        path: Directory containing subfolders with canopy_input.jsonl files.

    Returns:
        Combined list of all records from all found files.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {path}")

    records: list[DatasetRecord] = []
    for jsonl_path in sorted(path.rglob("canopy_input.jsonl")):
        records.extend(load_canopy_dataset(jsonl_path))

    return records


def split_dataset(
    records: list[DatasetRecord],
    train_frac: float = 0.7,
    eval_frac: float = 0.15,
    seed: int | None = None,
) -> tuple[list[DatasetRecord], list[DatasetRecord], list[DatasetRecord]]:
    """
    Split records randomly into train, eval, and test sets.

    Args:
        records: Combined list of records.
        train_frac: Fraction for train (default 0.7).
        eval_frac: Fraction for eval (default 0.15). Test gets the remainder.
        seed: Random seed for reproducibility.

    Returns:
        (train, eval, test) lists.
    """
    if seed is not None:
        random.seed(seed)

    shuffled = list(records)
    random.shuffle(shuffled)
    n = len(shuffled)

    train_end = int(n * train_frac)
    eval_end = train_end + int(n * eval_frac)

    train_data = shuffled[:train_end]
    eval_data = shuffled[train_end:eval_end]
    test_data = shuffled[eval_end:]

    return train_data, eval_data, test_data
