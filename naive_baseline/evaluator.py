"""Evaluate a policy string against a dataset, returning training MSE."""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from evolutionary_mimic.fitness import FAILURE_MSE, evaluate  # noqa: F401 — re-export
from evolutionary_mimic.dataset import DatasetRecord


def score(code: str, data: list[DatasetRecord]) -> float:
    """
    Evaluate a policy string against data records.

    Returns MSE on success, or FAILURE_MSE (inf) on parse/execution error.
    Does not raise; all errors are absorbed into the FAILURE_MSE sentinel.
    """
    mse, _ = evaluate(code, data)
    return mse
