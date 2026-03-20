"""Main evolutionary loop for the naive hill-climbing baseline."""

import csv
import json
import sys
from pathlib import Path
from typing import Optional

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from evolutionary_mimic.dataset import load_canopy_dataset_dir, split_dataset

from .evaluator import score
from .initializer import generate_seed
from .mutator import mutate

DEFAULT_MAX_GENERATIONS = 50
DEFAULT_PATIENCE = 5
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_SEED = 42


def run(
    dataset_dir: str | Path,
    model_name: str,
    output_dir: str | Path,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
    patience: int = DEFAULT_PATIENCE,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    split_seed: int = DEFAULT_SEED,
) -> None:
    """
    Run the naive hill-climbing evolutionary baseline.

    One candidate is produced per generation via LLM mutation of the
    current best. Selection is strict top-1: the candidate replaces the
    incumbent only if its training MSE is strictly lower. Terminates after
    max_generations or after patience consecutive non-improving generations.

    Outputs written to output_dir:
        log.csv          — per-generation row: generation, candidate_id,
                           train_mse, improved, stale_count
        best_policy.py   — incumbent policy code (overwritten on improvement)
        gen_N.py         — snapshot of the improving policy at generation N
        token_usage.json — cumulative and per-call token counts (overwritten
                           after every generation so it is always up to date)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.csv"
    token_path = output_dir / "token_usage.json"
    best_policy_path = output_dir / "best_policy.py"

    # ------------------------------------------------------------------ #
    # Load and split data                                                  #
    # ------------------------------------------------------------------ #
    print(f"Loading dataset from {dataset_dir} ...")
    records = load_canopy_dataset_dir(dataset_dir)
    if not records:
        raise RuntimeError(f"No records found under {dataset_dir}")
    print(f"Loaded {len(records)} records total.")

    # eval_frac=0.0 → only train split is used; remaining records are unused
    train_data, _, _ = split_dataset(
        records, train_frac=train_frac, eval_frac=0.0, seed=split_seed
    )
    print(f"Train split: {len(train_data)} records (seed={split_seed})")

    state_length = len(records[0]["state"])
    print(f"State length: {state_length}")

    # ------------------------------------------------------------------ #
    # Token tracking                                                       #
    # ------------------------------------------------------------------ #
    token_calls: list[dict] = []

    def _flush_tokens() -> None:
        total_in = sum(c["input_tokens"] for c in token_calls)
        total_out = sum(c["output_tokens"] for c in token_calls)
        payload = {
            "model": model_name,
            "calls": token_calls,
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "total_tokens": total_in + total_out,
        }
        token_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # CSV log                                                              #
    # ------------------------------------------------------------------ #
    csv_file = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["generation", "candidate_id", "train_mse", "improved", "stale_count"])

    def _log(gen: int, improved: bool, mse: float, stale: int) -> None:
        writer.writerow([gen, f"gen_{gen}", f"{mse:.8f}", int(improved), stale])
        csv_file.flush()

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    try:
        # Generation 0: seed
        print("Generating seed policy (gen 0) ...")
        seed_code, seed_stats = generate_seed(model_name, state_length)
        token_calls.append({"generation": 0, "call_type": "seed", **seed_stats})

        incumbent_code = seed_code
        incumbent_mse = score(seed_code, train_data)
        print(f"Gen 0 (seed): train_mse={incumbent_mse:.6f}")

        best_policy_path.write_text(incumbent_code, encoding="utf-8")
        (output_dir / "gen_0.py").write_text(incumbent_code, encoding="utf-8")
        _log(0, True, incumbent_mse, 0)
        _flush_tokens()

        stale_count = 0

        for gen in range(1, max_generations + 1):
            print(f"Gen {gen}: mutating (incumbent_mse={incumbent_mse:.6f}) ...")

            # -- LLM call ------------------------------------------------
            try:
                candidate_code, mut_stats = mutate(
                    incumbent_code, incumbent_mse, model_name, state_length
                )
                token_calls.append({"generation": gen, "call_type": "mutate", **mut_stats})
            except RuntimeError as e:
                print(f"  LLM call failed: {e}. Preserving incumbent.")
                stale_count += 1
                _log(gen, False, incumbent_mse, stale_count)
                _flush_tokens()
                if stale_count >= patience:
                    print(f"Stopping: {patience} consecutive non-improving generations.")
                    break
                continue

            # -- Parse failure -------------------------------------------
            if candidate_code is None:
                print(f"  Gen {gen}: LLM returned no parseable policy. Preserving incumbent.")
                stale_count += 1
                _log(gen, False, incumbent_mse, stale_count)
                _flush_tokens()
                if stale_count >= patience:
                    print(f"Stopping: {patience} consecutive non-improving generations.")
                    break
                continue

            # -- Evaluate and select -------------------------------------
            candidate_mse = score(candidate_code, train_data)
            improved = candidate_mse < incumbent_mse

            if improved:
                incumbent_code = candidate_code
                incumbent_mse = candidate_mse
                stale_count = 0
                best_policy_path.write_text(incumbent_code, encoding="utf-8")
                (output_dir / f"gen_{gen}.py").write_text(incumbent_code, encoding="utf-8")
                print(f"  Gen {gen}: improved → train_mse={incumbent_mse:.6f}")
            else:
                stale_count += 1
                print(
                    f"  Gen {gen}: no improvement "
                    f"(candidate={candidate_mse:.6f}, stale={stale_count}/{patience})"
                )

            _log(gen, improved, incumbent_mse, stale_count)
            _flush_tokens()

            if stale_count >= patience:
                print(f"Stopping: {patience} consecutive non-improving generations.")
                break

        print(f"\nDone. Best train_mse={incumbent_mse:.6f}")
        print(f"Best policy : {best_policy_path}")
        print(f"Log         : {log_path}")
        print(f"Token usage : {token_path}")

    finally:
        csv_file.close()
        _flush_tokens()
