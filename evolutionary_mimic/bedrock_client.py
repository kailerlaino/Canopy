"""AWS Bedrock client (deprecated). Use llm_client.py with LLMWrapper instead."""

import json
import re
from typing import Optional

import boto3

# Default model IDs for Bedrock (us-east-1)
DEFAULT_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
DEFAULT_REGION = "us-east-1"


def _invoke_bedrock(
    prompt: str,
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
    max_tokens: int = 4096,
) -> str:
    """
    Invoke Bedrock model with given prompt and return response text.

    Raises:
        Exception: On API or network errors.
    """
    client = boto3.client("bedrock-runtime", region_name=region)

    body = json.dumps(
        {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
    )

    response = client.invoke_model(
        body=body,
        modelId=model_id,
        contentType="application/json",
        accept="application/json",
    )

    response_body = json.loads(response["body"].read())
    # Extract text from Anthropic response format
    if "content" in response_body and response_body["content"]:
        return response_body["content"][0].get("text", "")
    return ""


def _extract_code_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown or raw text."""
    blocks: list[str] = []
    # Try markdown code blocks first
    pattern = r"```(?:python)?\s*\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    for m in matches:
        block = m.strip()
        if block and "def policy" in block:
            blocks.append(block)

    if blocks:
        return blocks

    # Fallback: look for def policy(...) ... until next def or end
    match = re.search(
        r"(def\s+policy\s*\([^)]*\)[^:]*:.*?)(?=\n\s*def\s|\n\n\n|\Z)",
        text,
        re.DOTALL,
    )
    if match:
        blocks.append(match.group(1).strip())

    return blocks


def generate_initial_population(
    n: int,
    model_id: str = DEFAULT_MODEL_ID,
    region: str = DEFAULT_REGION,
) -> list[str]:
    """
    Ask Bedrock to generate N initial Python policy functions.

    Returns list of code strings. May return fewer than N if parsing fails.
    """
    prompt = f"""Generate {n} distinct Python functions that approximate a congestion-control policy.

Input: state (list of 84 floats, 12 timesteps × 7 features: throughput, loss rate, inverseRTT, etc.)
Output: single float in [-1, 1]

Constraints:
- Use only numpy (as np) and math. No file I/O, no network, no imports.
- Function signature: def policy(state: list) -> float
- Each function must be named 'policy'

Return each function in a separate ```python code block. No explanations, only code blocks."""

    try:
        text = _invoke_bedrock(prompt, model_id=model_id, region=region, max_tokens=8192)
    except Exception as e:
        raise RuntimeError(f"Bedrock API error: {e}") from e

    blocks = _extract_code_blocks(text)
    # Ensure each block has full def policy
    result: list[str] = []
    for block in blocks:
        if "def policy" in block and "state" in block:
            result.append(block)
        if len(result) >= n:
            break

    return result[:n]


def mutate(code: str, mse: float, model_id: str = DEFAULT_MODEL_ID, region: str = DEFAULT_REGION) -> Optional[str]:
    """
    Ask Bedrock to improve the given policy code to reduce MSE.

    Returns improved code string, or None if parsing fails.
    """
    prompt = f"""Here is a Python function and its MSE on a dataset of 1504 (state, action) pairs.
Current MSE: {mse:.6f}

Function:
```python
{code}
```

Improve it to reduce MSE. Keep the same signature: def policy(state: list) -> float
Use only numpy (np) and math. No file I/O, no network.
Return ONLY the improved function in a ```python code block, no explanations."""

    try:
        text = _invoke_bedrock(prompt, model_id=model_id, region=region, max_tokens=4096)
    except Exception as e:
        raise RuntimeError(f"Bedrock API error: {e}") from e

    blocks = _extract_code_blocks(text)
    for block in blocks:
        if "def policy" in block and "state" in block:
            return block

    return None
