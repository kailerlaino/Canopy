"""CLI entrypoint for evolutionary mimicry."""

import argparse
from pathlib import Path

from .config import (
    DEFAULT_CHECKPOINT_FILE,
    DEFAULT_DATASET_PATH,
    DEFAULT_METRICS_FILE,
    DEFAULT_MODEL_TIER,
    DEFAULT_OUTPUT_FILE,
    DEFAULT_TOKEN_USAGE_FILE,
    EARLY_STOP_PATIENCE,
    EVAL_FRAC,
    FRESH_INJECTION_PER_GEN,
    MAX_GENERATIONS,
    MUTATIONS_PER_SURVIVOR,
    POPULATION_SIZE,
    SPLIT_SEED,
    SURVIVORS_PER_GEN,
    TRAIN_FRAC,
)
from .dataset import load_canopy_dataset, load_canopy_dataset_dir, split_dataset
from .evolution import run_evolution


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evolve Python functions to mimic neural network policy from canopy_input.jsonl"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to single canopy_input.jsonl file",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="Folder containing subfolders with canopy_input.jsonl. Pools all records, splits train/eval/test, runs one evolution for a single generalized policy.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory when using --dataset-dir. Default: same as dataset-dir.",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=TRAIN_FRAC,
        help=f"Fraction of data for training (default: {TRAIN_FRAC})",
    )
    parser.add_argument(
        "--eval-frac",
        type=float,
        default=EVAL_FRAC,
        help=f"Fraction of data for eval (default: {EVAL_FRAC})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_SEED,
        help=f"Random seed for train/eval/test split (default: {SPLIT_SEED})",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=MAX_GENERATIONS,
        help=f"Max generations (default: {MAX_GENERATIONS})",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=POPULATION_SIZE,
        help=f"Population size (default: {POPULATION_SIZE})",
    )
    parser.add_argument(
        "--survivors",
        type=int,
        default=SURVIVORS_PER_GEN,
        help=f"Survivors per generation (default: {SURVIVORS_PER_GEN})",
    )
    parser.add_argument(
        "--mutations",
        type=int,
        default=MUTATIONS_PER_SURVIVOR,
        help=f"Mutations per survivor (default: {MUTATIONS_PER_SURVIVOR})",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOP_PATIENCE,
        help=f"Early stop patience (default: {EARLY_STOP_PATIENCE})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help=f"Output file for best policy (default: {DEFAULT_OUTPUT_FILE})",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=DEFAULT_METRICS_FILE,
        help=f"Output file for metrics JSON (default: {DEFAULT_METRICS_FILE})",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name (e.g. nova-lite, gemini-2.0). Overrides --model-tier.",
    )
    parser.add_argument(
        "--model-tier",
        type=str,
        choices=["cheap", "medium", "expensive"],
        default=DEFAULT_MODEL_TIER,
        help=f"Model tier for cost control (default: {DEFAULT_MODEL_TIER})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start a new run from scratch; ignore existing checkpoint even if present",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT_FILE,
        help=f"Checkpoint file path (default: {DEFAULT_CHECKPOINT_FILE})",
    )
    parser.add_argument(
        "--token-usage",
        type=str,
        default=DEFAULT_TOKEN_USAGE_FILE,
        help=f"Token usage output file (default: {DEFAULT_TOKEN_USAGE_FILE})",
    )
    parser.add_argument(
        "--best-policies-dir",
        type=str,
        default=None,
        help="Directory to save best policy per generation (e.g. best_policies/gen_0.py). Default: <output-dir>/best_policies when using --dataset-dir.",
    )

    args = parser.parse_args()

    # Load and split data
    if args.dataset_dir:
        dataset_dir = Path(args.dataset_dir)
        if not dataset_dir.is_dir():
            raise SystemExit(f"Dataset directory not found: {dataset_dir}")

        records = load_canopy_dataset_dir(dataset_dir)
        if not records:
            raise SystemExit(f"No records found under {dataset_dir}")

        output_dir = Path(args.output_dir) if args.output_dir else dataset_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "best_policy.py"
        metrics_path = output_dir / "evolution_metrics.json"
        checkpoint_path = output_dir / "evolution_checkpoint.json"
        token_usage_path = output_dir / "token_usage.json"
        best_policies_dir = args.best_policies_dir or str(output_dir / "best_policies")
        resume_path = None if args.fresh else (args.resume if args.resume is not None else (str(checkpoint_path) if checkpoint_path.exists() else None))

        print(f"Loaded {len(records)} records from dataset dir. Splitting train/eval/test.")
    else:
        dataset_path = args.dataset or str(DEFAULT_DATASET_PATH)
        records = load_canopy_dataset(dataset_path)
        if not records:
            raise SystemExit(f"No records in {dataset_path}")

        output_path = args.output
        metrics_path = args.metrics
        checkpoint_path = args.checkpoint
        token_usage_path = args.token_usage
        best_policies_dir = args.best_policies_dir
        resume_path = None if args.fresh else args.resume

        print(f"Loaded {len(records)} records. Splitting train/eval/test.")

    train_data, eval_data, test_data = split_dataset(
        records,
        train_frac=args.train_frac,
        eval_frac=args.eval_frac,
        seed=args.seed,
    )
    print(f"Train: {len(train_data)}, Eval: {len(eval_data)}, Test: {len(test_data)}")

    run_evolution(
        train_data=train_data,
        eval_data=eval_data,
        test_data=test_data,
        population_size=args.population,
        survivors_per_gen=args.survivors,
        mutations_per_survivor=args.mutations,
        max_generations=args.generations,
        early_stop_patience=args.patience,
        fresh_injection_per_gen=FRESH_INJECTION_PER_GEN,
        model_name=args.model,
        model_tier=args.model_tier,
        output_path=output_path,
        metrics_path=metrics_path,
        checkpoint_path=checkpoint_path,
        token_usage_path=token_usage_path,
        resume_path=resume_path,
        best_policies_dir=best_policies_dir,
        split_seed=args.seed,
    )


if __name__ == "__main__":
    main()
