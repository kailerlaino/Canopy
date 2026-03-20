"""Mutate the incumbent policy via a single LLM call."""

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


def mutate(
    code: str,
    mse: float,
    model_name: str,
    state_length: int,
) -> tuple[Optional[str], dict]:
    """
    Ask the LLM to improve the incumbent policy to reduce MSE.

    The prompt passes only the code and its numeric MSE — no qualitative
    feedback, no verbal gradients. This interface is the sole point of
    contact between the loop and the LLM prompt strategy; swap the prompt
    here to test richer feedback mechanisms without touching loop.py.

    Returns:
        (code, stats): improved policy code (or None if the response could
                       not be parsed) and token usage dict
                       {input_tokens, output_tokens, total_tokens}.
    Raises:
        RuntimeError: if the LLM call itself fails after all retries.
    """
    prompt = (
        f"Current MSE: {mse:.6f}\n\n"
        f"```python\n{code}\n```\n\n"
        f"Rewrite this function to reduce the MSE on a congestion control dataset.\n"
        f"Signature: def policy(state: list) -> float\n"
        f"Output: single float in [-1, 1]\n"
        f"State is a flat list of {state_length} floats. Use len(state) for sizing.\n"
        f"Allowed: numpy (as np) and math only. No file I/O, no network calls.\n\n"
        f"Return ONLY the improved function in a ```python code block. No explanation."
    )

    last_error: Optional[Exception] = None
    for attempt in range(MAX_RETRIES):
        try:
            wrapper = get_wrapper(model_name)
            text = wrapper.send(prompt)
            stats = wrapper._stats()
            # Return even if extraction yields None — caller handles that case
            return _extract_policy_code(text), stats
        except Exception as e:
            last_error = e

        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF_BASE_SEC * (3 ** attempt))

    raise RuntimeError(
        f"LLM call failed after {MAX_RETRIES} attempts: {last_error}"
    ) from last_error
