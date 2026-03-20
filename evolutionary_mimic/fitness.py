"""Fitness evaluation: MSE between evolved policy output and target action."""

import math
import re
from typing import Callable

import numpy as np

from .dataset import DatasetRecord

# Infinite MSE for invalid/unsafe code
FAILURE_MSE = float("inf")


def _create_safe_globals() -> dict:
    """Create minimal globals for exec: only numpy and math, no I/O or imports."""
    safe_builtins = {
        "abs": abs,
        "all": all,
        "any": any,
        "bool": bool,
        "dict": dict,
        "enumerate": enumerate,
        "float": float,
        "int": int,
        "len": len,
        "list": list,
        "map": map,
        "max": max,
        "min": min,
        "pow": pow,
        "range": range,
        "round": round,
        "set": set,
        "sorted": sorted,
        "sum": sum,
        "tuple": tuple,
        "zip": zip,
        "True": True,
        "False": False,
        "None": None,
    }
    return {
        "__builtins__": safe_builtins,
        "np": np,
        "math": math,
    }


def _preprocess_code(code: str) -> str:
    """
    Remove import statements that would fail in our sandbox.
    np and math are already in globals; LLMs often add these anyway.
    """
    # Remove common import lines that we pre-inject in globals
    patterns = [
        r"^\s*import\s+numpy\s+as\s+np\s*\n",
        r"^\s*import\s+numpy\s*\n",
        r"^\s*import\s+math\s*\n",
        r"^\s*from\s+numpy\s+import\s+[^\n]+\n",
        r"^\s*from\s+math\s+import\s+[^\n]+\n",
    ]
    result = code
    for p in patterns:
        result = re.sub(p, "", result, flags=re.MULTILINE)
    # Strip stray markdown backticks LLMs sometimes append
    result = re.sub(r"\n*```+\s*\n*$", "", result)
    return result.strip()


def _extract_policy_from_code(code: str) -> Callable[[list[float]], float] | None:
    """
    Execute code in restricted env and return the policy function.

    Expects code to define a function named 'policy' with signature:
        def policy(state: list) -> float

    Returns None if execution fails or no valid policy is found.
    """
    code = _preprocess_code(code)
    globals_dict = _create_safe_globals()
    try:
        exec(code, globals_dict)
        policy = globals_dict.get("policy")
        if policy is None or not callable(policy):
            return None
        return policy
    except Exception:
        return None


def evaluate(code: str, dataset: list[DatasetRecord]) -> tuple[float, bool]:
    """
    Evaluate evolved code against dataset using MSE.

    Args:
        code: Python code string that defines `def policy(state: list) -> float`.
        dataset: List of {state, action_no_noise} records.

    Returns:
        (mse, success): MSE value (or FAILURE_MSE on error), and whether evaluation succeeded.
    """
    policy = _extract_policy_from_code(code)
    if policy is None:
        return FAILURE_MSE, False

    squared_errors: list[float] = []
    for rec in dataset:
        state = rec["state"]
        target = rec["action_no_noise"][0]
        try:
            pred = policy(state)
            if not isinstance(pred, (int, float)):
                pred = float(pred)
            # Clamp to [-1, 1] for fair comparison (target is in that range)
            pred = max(-1.0, min(1.0, float(pred)))
            squared_errors.append((pred - target) ** 2)
        except Exception:
            return FAILURE_MSE, False

    if not squared_errors:
        return FAILURE_MSE, False

    mse = sum(squared_errors) / len(squared_errors)
    return mse, True
