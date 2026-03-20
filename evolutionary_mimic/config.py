"""Configuration for evolutionary mimicry."""

from pathlib import Path

# Default dataset path (wired12)
DEFAULT_DATASET_PATH = Path(__file__).resolve().parent.parent / "users" / "kailer" / "eval_results" / "robustness" / "run1" / "dataset" / "wired12" / "canopy_input.jsonl"

# Default dataset directory (for multi-dataset runs)
DEFAULT_DATASET_DIR = Path(__file__).resolve().parent.parent / "users" / "kailer" / "eval_results" / "robustness" / "run1" / "dataset"

# Evolutionary parameters (tuned for better convergence)
POPULATION_SIZE = 25
SURVIVORS_PER_GEN = 8
MUTATIONS_PER_SURVIVOR = 3
MAX_GENERATIONS = 40
EARLY_STOP_PATIENCE = 10  # Stop if no improvement for this many generations
FRESH_INJECTION_PER_GEN = 2  # New LLM-generated individuals per generation

# Bedrock
BEDROCK_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
BEDROCK_REGION = "us-east-1"

# Output
DEFAULT_OUTPUT_FILE = "best_policy.py"
DEFAULT_METRICS_FILE = "evolution_metrics.json"
DEFAULT_CHECKPOINT_FILE = "evolution_checkpoint.json"
DEFAULT_TOKEN_USAGE_FILE = "token_usage.json"

# Model tiers (cheaper first for testing)
MODEL_TIERS = {
    "cheap": ["nova-lite", "gemini-2.0", "gpt-4o-mini"],
    "medium": ["nova-pro", "claude-haiku4.5", "gpt-5-nano"],
    "expensive": ["claude-opus4.5", "nova-premier", "gemini-2.5"],
}
DEFAULT_MODEL_TIER = "cheap"

# Train/eval/test split
TRAIN_FRAC = 0.7
EVAL_FRAC = 0.15
TEST_FRAC = 0.15
SPLIT_SEED = 42

# Retry
RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE_SEC = 5  # 5s, 15s, 45s
