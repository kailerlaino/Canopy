"""CLI entry point for the naive evolutionary baseline."""

from .loop import (
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_FRAC,
    run,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Naive hill-climbing evolutionary baseline: single-survivor, "
            "single-mutation policy mimicry via LLM rewrite."
        )
    )
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Directory containing subdirectories with canopy_input.jsonl files.",
    )
    parser.add_argument(
        "--model",
        required=True,
        help="LLMWrapper model key (e.g. nova-lite, claude-haiku4.5).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write log.csv, best_policy.py, token_usage.json.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help=f"Max generations (default: {DEFAULT_MAX_GENERATIONS}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Stop after this many consecutive non-improving generations (default: {DEFAULT_PATIENCE}).",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=DEFAULT_TRAIN_FRAC,
        help=f"Fraction of records used for training (default: {DEFAULT_TRAIN_FRAC}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed for train/holdout split (default: {DEFAULT_SEED}).",
    )

    args = parser.parse_args()

    run(
        dataset_dir=args.dataset_dir,
        model_name=args.model,
        output_dir=args.output_dir,
        max_generations=args.generations,
        patience=args.patience,
        train_frac=args.train_frac,
        split_seed=args.seed,
    )


if __name__ == "__main__":
    main()
