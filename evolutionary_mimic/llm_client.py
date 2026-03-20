"""LLM client using LLMWrapper for generating and mutating policy code."""

import re
import sys
import time
from pathlib import Path
from typing import Optional

# Ensure ldos root is on path for LLMWrapper import
_ldos_root = Path(__file__).resolve().parent.parent
if str(_ldos_root) not in sys.path:
    sys.path.insert(0, str(_ldos_root))

from LLMWrapper import get_wrapper

from .config import RETRY_ATTEMPTS, RETRY_BACKOFF_BASE_SEC


def _normalize_stats(raw: dict) -> dict:
    """Normalize wrapper stats to {input_tokens, output_tokens, total_tokens}."""
    # Bedrock: input_tokens, output_tokens, total_tokens
    if "input_tokens" in raw and "output_tokens" in raw:
        return {
            "input_tokens": raw["input_tokens"],
            "output_tokens": raw["output_tokens"],
            "total_tokens": raw.get("total_tokens", raw["input_tokens"] + raw["output_tokens"]),
        }
    # Gemini: prompt_total or total, output or candidates_token_count
    if "prompt_total" in raw or "total" in raw:
        inp = raw.get("input_tokens") or raw.get("prompt_total") or raw.get("total", 0)
        out = raw.get("output_tokens") or raw.get("output") or raw.get("candidates_token_count", 0)
        return {
            "input_tokens": inp,
            "output_tokens": out,
            "total_tokens": raw.get("total_tokens") or raw.get("total") or (inp + out),
        }
    return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}


def _invoke_with_retry(model_name: str, prompt: str, call_type: str) -> tuple[str, dict]:
    """
    Invoke LLM with retry and exponential backoff.
    Returns (response_text, normalized_stats).
    Raises on final failure after retries.
    """
    last_error = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            wrapper = get_wrapper(model_name)
            text = wrapper.send(prompt)
            raw = wrapper._stats()
            stats = _normalize_stats(raw)
            # print(f"[LLM DEBUG] call_type={call_type} len={len(text)}")
            # print(f"[LLM DEBUG] raw response:\n---\n{text}\n---")
            return text, stats
        except Exception as e:
            last_error = e
            if attempt < RETRY_ATTEMPTS - 1:
                delay = RETRY_BACKOFF_BASE_SEC * (3 ** attempt)
                time.sleep(delay)
            else:
                raise RuntimeError(f"LLM call failed after {RETRY_ATTEMPTS} attempts: {e}") from last_error
    raise RuntimeError(f"LLM call failed: {last_error}") from last_error


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown or raw text."""
    blocks: list[str] = []
    # Markdown: ```python or ```, optional space/newline, then code until ```
    pattern = r"```(?:python)?\s*\n?(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        block = m.strip()
        if block and "def policy" in block:
            blocks.append(block)

    if blocks:
        return blocks

    # Raw code fallback: from "def policy" until double newline + non-indented line, or EOF
    match = re.search(
        r"(def\s+policy\s*\([^)]*\)[^:]*:(?:\n(?:[ \t]+.*| *$))*)",
        text,
        re.MULTILINE,
    )
    if match:
        block = match.group(1).rstrip()
        if block and "state" in block:
            blocks.append(block)
    return blocks


def generate_initial_population(
    n: int,
    model_name: str,
    state_length: int = 84,
) -> tuple[list[str], list[dict]]:
    """
    Ask LLM to generate N initial Python policy functions.

    Returns (list of code strings, list of stats dicts per call).
    """
    prompt = f"""Generate {n} distinct Python functions that approximate a congestion-control policy.

Input: state - a FLAT list of {state_length} floats (timesteps × 7 features: throughput, loss rate, inverseRTT, etc.)
Output: single float in [-1, 1]

CRITICAL: state is 1D. Use np.array(state).flatten() and len(state). Do NOT use 2D indexing like [:, 0].
Do NOT hardcode 84 or 12 - use len(state) for any array sizing.

Constraints:
- Use only numpy (as np) and math. No file I/O, no network, no imports.
- Function signature: def policy(state: list) -> float
- Each function must be named 'policy'

Return each function in a separate ```python code block. No explanations, only code blocks."""

    text, stats = _invoke_with_retry(model_name, prompt, "generate_initial")
    blocks = _extract_code_blocks(text)
    result: list[str] = []
    for block in blocks:
        if "def policy" in block and "state" in block:
            result.append(block)
        if len(result) >= n:
            break

    return result[:n], [stats]


def mutate(
    code: str,
    mse: float,
    model_name: str,
    state_length: int = 84,
) -> tuple[Optional[str], dict]:
    """
    Ask LLM to improve the given policy code to reduce MSE.

    Returns (improved code string or None, stats dict).
    """
    prompt = f"""Here is a Python function and its MSE on a dataset of (state, action) pairs.
Current MSE: {mse:.6f}

Function:
```python
{code}
```

Improve it to reduce MSE. Keep the same signature: def policy(state: list) -> float
CRITICAL: state is a FLAT list of {state_length} floats. Use np.array(state).flatten() and len(state). Do NOT use 2D indexing like [:, 0].
Use only numpy (np) and math. No file I/O, no network.
Return ONLY the improved function in a ```python code block, no explanations."""

    text, stats = _invoke_with_retry(model_name, prompt, "mutate")
    blocks = _extract_code_blocks(text)
    for block in blocks:
        if "def policy" in block and "state" in block:
            return block, stats

    return None, stats
