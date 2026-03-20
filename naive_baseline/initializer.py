"""Generate the seed policy for the naive evolutionary baseline."""

import re
import sys
import time
from pathlib import Path
from typing import Optional

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from LLMWrapper import get_wrapper

MAX_RETRIES = 3
RETRY_BACKOFF_BASE_SEC = 5


def _extract_policy_code(text: str) -> Optional[str]:
    """Extract the first ```python block containing 'def policy', or fall back to raw def."""
    for match in re.finditer(r"```(?:python)?\s*\n?(.*?)```", text, re.DOTALL):
        block = match.group(1).strip()
        if "def policy" in block:
            return block
    match = re.search(
        r"(def\s+policy\s*\([^)]*\)[^:]*:(?:\n(?:[ \t]+.*| *$))*)",
        text,
        re.MULTILINE,
    )
    if match:
        block = match.group(1).rstrip()
        if "state" in block:
            return block
    return None


def generate_seed(model_name: str, state_length: int) -> tuple[str, dict]:
    """
    Generate an initial policy function via one LLM call.

    Returns:
        (code, stats): policy code string and token usage dict
                       {input_tokens, output_tokens, total_tokens}.
    Raises:
        RuntimeError: if all retries are exhausted without a valid policy.
    """
    n_timesteps = state_length // 7
    prompt = (
        f"Write a Python function that approximates a network congestion control policy.\n\n"
        f"The function receives a flat list of {state_length} floats representing "
        f"{n_timesteps} timesteps × 7 features per timestep:\n"
        f"  indices [0, 7, 14, ...] = throughput\n"
        f"  indices [1, 8, 15, ...] = loss rate\n"
        f"  indices [2, 9, 16, ...] = inverse RTT\n"
        f"  indices [3, 10, 17, ...] = jitter\n"
        f"  indices [4, 11, 18, ...] = bandwidth\n"
        f"  indices [5, 12, 19, ...] = error rate\n"
        f"  indices [6, 13, 20, ...] = queue length\n\n"
        f"Signature: def policy(state: list) -> float\n"
        f"Output: a single float in [-1, 1]\n"
        f"Allowed: numpy (as np) and math only. No file I/O, no network calls.\n"
        f"Use len(state) for sizing — do NOT hardcode {state_length}.\n\n"
        f"Return ONLY the function in a ```python code block. No explanation."
    )

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            wrapper = get_wrapper(model_name)
            text = wrapper.send(prompt)
            stats = wrapper._stats()
            code = _extract_policy_code(text)
            if code is not None:
                return code, stats
            # Parsed fine but no policy found — count as a soft failure and retry
            last_error = ValueError("LLM response contained no valid policy function")
        except Exception as e:
            last_error = e

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_BASE_SEC * (3 ** attempt))

    raise RuntimeError(
        f"Failed to generate a valid seed policy after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    ) from last_error
