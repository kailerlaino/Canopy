"""CLI entry point for the island-based evolutionary system."""

from .loop import (
    DEFAULT_DIVERSITY_THRESHOLD,
    DEFAULT_MAX_GENERATIONS,
    DEFAULT_MIGRATION_INTERVAL,
    DEFAULT_NOVELTY_THRESHOLD,
    DEFAULT_NUM_ISLANDS,
    DEFAULT_PATIENCE,
    DEFAULT_SEED,
    DEFAULT_TRAIN_FRAC,
    run,
)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Island-based evolutionary policy mimicry: multiple isolated populations "
            "with periodic migration, novelty rejection, and dynamic island spawning."
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
        help="Directory to write log.csv, best_policy.py, island_N_best.py, token_usage.json.",
    )
    parser.add_argument(
        "--generations",
        type=int,
        default=DEFAULT_MAX_GENERATIONS,
        help=f"Max generations (default: {DEFAULT_MAX_GENERATIONS}).",
    )
    parser.add_argument(
        "--islands",
        type=int,
        default=DEFAULT_NUM_ISLANDS,
        help=f"Number of initial islands (default: {DEFAULT_NUM_ISLANDS}).",
    )
    parser.add_argument(
        "--migration-interval",
        type=int,
        default=DEFAULT_MIGRATION_INTERVAL,
        help=f"Migrate every N generations (default: {DEFAULT_MIGRATION_INTERVAL}).",
    )
    parser.add_argument(
        "--novelty-threshold",
        type=float,
        default=DEFAULT_NOVELTY_THRESHOLD,
        help=f"Cosine similarity threshold for novelty rejection (default: {DEFAULT_NOVELTY_THRESHOLD}).",
    )
    parser.add_argument(
        "--diversity-threshold",
        type=float,
        default=DEFAULT_DIVERSITY_THRESHOLD,
        help=f"Mean pairwise similarity above which a new island is spawned (default: {DEFAULT_DIVERSITY_THRESHOLD}).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
        help=f"Stale generations before an island is retired (default: {DEFAULT_PATIENCE}).",
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
        num_islands=args.islands,
        migration_interval=args.migration_interval,
        novelty_threshold=args.novelty_threshold,
        diversity_threshold=args.diversity_threshold,
        patience=args.patience,
        train_frac=args.train_frac,
        split_seed=args.seed,
    )


if __name__ == "__main__":
    main()
