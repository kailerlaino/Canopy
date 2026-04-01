"""Multi-island evolutionary loop.

Orchestrates K islands evolving in parallel (sequentially in this
implementation — parallelism can be added later via concurrent.futures
without changing the public run() signature).

Per-generation CSV columns (superset of naive_baseline log.csv):
    generation, island_id, candidate_id, train_mse, global_best_mse,
    improved, stale_count, novelty_rejected

Dynamic island spawning:
    When mean pairwise cosine similarity across incumbents exceeds
    `diversity_threshold`, a fresh island is seeded from the LLM.
    Islands that stay stale for `patience` consecutive generations and
    are not rescued by migration are retired (min floor: `num_islands`).
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from evolutionary_mimic.dataset import load_canopy_dataset_dir, split_dataset
from naive_baseline.initializer import generate_seed

from .evaluator import FAILURE_MSE, score
from .island import Island, step_island
from .migration import maybe_migrate
from .novelty import NoveltyFilter, mean_pairwise_similarity

DEFAULT_MAX_GENERATIONS = 50
DEFAULT_NUM_ISLANDS = 4
DEFAULT_MIGRATION_INTERVAL = 10
DEFAULT_NOVELTY_THRESHOLD = 0.95
DEFAULT_DIVERSITY_THRESHOLD = 0.85
DEFAULT_PATIENCE = 5
DEFAULT_TRAIN_FRAC = 0.7
DEFAULT_SEED = 42


def run(
    dataset_dir: str | Path,
    model_name: str,
    output_dir: str | Path,
    max_generations: int = DEFAULT_MAX_GENERATIONS,
    num_islands: int = DEFAULT_NUM_ISLANDS,
    migration_interval: int = DEFAULT_MIGRATION_INTERVAL,
    novelty_threshold: float = DEFAULT_NOVELTY_THRESHOLD,
    diversity_threshold: float = DEFAULT_DIVERSITY_THRESHOLD,
    patience: int = DEFAULT_PATIENCE,
    train_frac: float = DEFAULT_TRAIN_FRAC,
    split_seed: int = DEFAULT_SEED,
) -> None:
    """Run the island-based evolutionary system.

    Outputs written to output_dir:
        log.csv            — per-generation, per-island rows
        best_policy.py     — global best policy (overwritten on improvement)
        island_N_best.py   — best incumbent per island at termination
        token_usage.json   — cumulative + per-call token counts
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "log.csv"
    token_path = output_dir / "token_usage.json"
    best_policy_path = output_dir / "best_policy.py"

    # ------------------------------------------------------------------ #
    # Data                                                                 #
    # ------------------------------------------------------------------ #
    print(f"Loading dataset from {dataset_dir} ...")
    records = load_canopy_dataset_dir(dataset_dir)
    if not records:
        raise RuntimeError(f"No records found under {dataset_dir}")
    print(f"Loaded {len(records)} records total.")

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
        total_in = sum(c.get("input_tokens", 0) for c in token_calls)
        total_out = sum(c.get("output_tokens", 0) for c in token_calls)
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
    writer.writerow([
        "generation", "island_id", "candidate_id",
        "train_mse", "global_best_mse",
        "improved", "stale_count", "novelty_rejected",
    ])

    def _log(
        gen: int,
        island_id: int,
        train_mse: float,
        global_best: float,
        improved: bool,
        stale: int,
        novelty_rejected: bool,
    ) -> None:
        mse_str = f"{train_mse:.8f}" if train_mse != FAILURE_MSE else "inf"
        gbest_str = f"{global_best:.8f}" if global_best != FAILURE_MSE else "inf"
        writer.writerow([
            gen,
            island_id,
            f"gen{gen}_isl{island_id}",
            mse_str,
            gbest_str,
            int(improved),
            stale,
            int(novelty_rejected),
        ])
        csv_file.flush()

    # ------------------------------------------------------------------ #
    # Helpers                                                              #
    # ------------------------------------------------------------------ #
    next_island_id = [num_islands]  # mutable counter via list

    def _spawn_island(gen: int) -> Island:
        iid = next_island_id[0]
        next_island_id[0] += 1
        print(f"  Spawning new island {iid} (gen {gen}) — diversity collapse detected.")
        seed_code, seed_stats = generate_seed(model_name, state_length)
        token_calls.append({"generation": gen, "call_type": "seed_spawn", "island_id": iid, **seed_stats})
        seed_mse = score(seed_code, train_data)
        novelty_filter.register(seed_code)
        return Island(
            island_id=iid,
            incumbent_code=seed_code,
            incumbent_mse=seed_mse,
            stale_count=0,
            generation_born=gen,
        )

    # ------------------------------------------------------------------ #
    # Initialise islands (generation 0)                                    #
    # ------------------------------------------------------------------ #
    novelty_filter = NoveltyFilter(threshold=novelty_threshold)
    islands: list[Island] = []

    print(f"Seeding {num_islands} islands ...")
    for i in range(num_islands):
        print(f"  Island {i}: generating seed ...")
        seed_code, seed_stats = generate_seed(model_name, state_length)
        token_calls.append({"generation": 0, "call_type": "seed", "island_id": i, **seed_stats})
        seed_mse = score(seed_code, train_data)
        novelty_filter.register(seed_code)
        islands.append(Island(
            island_id=i,
            incumbent_code=seed_code,
            incumbent_mse=seed_mse,
            generation_born=0,
        ))
        print(f"    Island {i} seed mse={seed_mse:.6f}")

    global_best_mse = min(isl.incumbent_mse for isl in islands)
    global_best_code = min(islands, key=lambda isl: isl.incumbent_mse).incumbent_code
    best_policy_path.write_text(global_best_code, encoding="utf-8")

    # Log gen-0 seeds
    for isl in islands:
        _log(0, isl.island_id, isl.incumbent_mse, global_best_mse, True, 0, False)
    _flush_tokens()

    # ------------------------------------------------------------------ #
    # Main loop                                                            #
    # ------------------------------------------------------------------ #
    try:
        for gen in range(1, max_generations + 1):
            novelty_filter.reset_gen_counters()
            print(f"\nGen {gen} — {len(islands)} islands, global_best={global_best_mse:.6f}")

            # -- Step each island ----------------------------------------
            updated_islands: list[Island] = []
            for isl in islands:
                result = step_island(
                    isl, model_name, state_length, train_data, novelty_filter, patience
                )
                if result.token_stats:
                    token_calls.append({
                        "generation": gen,
                        "call_type": "mutate",
                        "island_id": isl.island_id,
                        **result.token_stats,
                    })

                # Update global best
                if result.island.incumbent_mse < global_best_mse:
                    global_best_mse = result.island.incumbent_mse
                    global_best_code = result.island.incumbent_code
                    best_policy_path.write_text(global_best_code, encoding="utf-8")
                    print(f"  [island {isl.island_id}] new global best: {global_best_mse:.6f}")

                _log(
                    gen,
                    result.island.island_id,
                    result.island.incumbent_mse,
                    global_best_mse,
                    result.improved,
                    result.island.stale_count,
                    result.novelty_rejected,
                )
                updated_islands.append(result.island)

            islands = updated_islands

            # -- Retire dead islands (stale >= patience), keep floor -----
            alive = [isl for isl in islands if isl.stale_count < patience]
            retired = [isl for isl in islands if isl.stale_count >= patience]
            if retired and len(alive) >= num_islands:
                for r in retired:
                    print(f"  Retiring island {r.island_id} (stale={r.stale_count})")
                islands = alive
            # If retiring would drop below floor, keep the least-stale ones
            elif retired:
                islands.sort(key=lambda isl: isl.stale_count)
                islands = islands[:max(num_islands, len(alive))]

            # -- Migration -----------------------------------------------
            islands = maybe_migrate(islands, gen, migration_interval, novelty_filter)

            # -- Dynamic spawning (diversity collapse) -------------------
            incumbents = [isl.incumbent_code for isl in islands]
            similarity = mean_pairwise_similarity(incumbents)
            max_islands = num_islands * 2
            if similarity >= diversity_threshold and len(islands) < max_islands:
                print(f"  Diversity collapse (sim={similarity:.3f} >= {diversity_threshold})")
                try:
                    new_isl = _spawn_island(gen)
                    islands.append(new_isl)
                    _log(gen, new_isl.island_id, new_isl.incumbent_mse, global_best_mse, True, 0, False)
                except RuntimeError as e:
                    print(f"  Spawn failed: {e}")

            rej_rate = novelty_filter.rejection_rate
            print(
                f"  novelty_rejection_rate={rej_rate:.2%}, "
                f"archive_size={novelty_filter.archive_size}"
            )
            _flush_tokens()

        print(f"\nDone. global_best_mse={global_best_mse:.6f}")
        print(f"Best policy : {best_policy_path}")
        print(f"Log         : {log_path}")
        print(f"Token usage : {token_path}")

        # Save per-island best policies
        for isl in islands:
            p = output_dir / f"island_{isl.island_id}_best.py"
            p.write_text(isl.incumbent_code, encoding="utf-8")

    finally:
        csv_file.close()
        _flush_tokens()
