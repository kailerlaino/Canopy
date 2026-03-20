"""Main evolutionary loop for policy mimicry."""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import (
    DEFAULT_MODEL_TIER,
    EARLY_STOP_PATIENCE,
    FRESH_INJECTION_PER_GEN,
    MAX_GENERATIONS,
    MODEL_TIERS,
    MUTATIONS_PER_SURVIVOR,
    POPULATION_SIZE,
    SURVIVORS_PER_GEN,
)
from .dataset import DatasetRecord
from .fitness import FAILURE_MSE, evaluate
from .llm_client import generate_initial_population, mutate


def _save_checkpoint(
    path: Path,
    generation: int,
    population: list[tuple[str, float]],
    best_code: str,
    best_mse_eval: float,
    best_generation: int,
    metrics_history: list[dict],
    config_snapshot: dict,
    best_mse_train: float | None = None,
) -> None:
    data = {
        "generation": generation,
        "population": [{"code": c, "mse": m} for c, m in population],
        "best_code": best_code,
        "best_mse": best_mse_eval,
        "best_mse_eval": best_mse_eval,
        "best_mse_train": best_mse_train,
        "best_generation": best_generation,
        "metrics_history": metrics_history,
        "config_snapshot": config_snapshot,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _load_checkpoint(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_token_usage(path: Path, run_id: str, model: str, calls: list[dict]) -> None:
    total_in = sum(c.get("input_tokens", 0) for c in calls)
    total_out = sum(c.get("output_tokens", 0) for c in calls)
    data = {
        "run_id": run_id,
        "model": model,
        "total_input_tokens": total_in,
        "total_output_tokens": total_out,
        "total_tokens": total_in + total_out,
        "calls": calls,
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_evolution(
    train_data: list[DatasetRecord],
    eval_data: list[DatasetRecord],
    test_data: list[DatasetRecord],
    population_size: int = POPULATION_SIZE,
    survivors_per_gen: int = SURVIVORS_PER_GEN,
    mutations_per_survivor: int = MUTATIONS_PER_SURVIVOR,
    max_generations: int = MAX_GENERATIONS,
    early_stop_patience: int = EARLY_STOP_PATIENCE,
    fresh_injection_per_gen: int = FRESH_INJECTION_PER_GEN,
    model_name: Optional[str] = None,
    model_tier: str = DEFAULT_MODEL_TIER,
    output_path: Optional[str | Path] = None,
    metrics_path: Optional[str | Path] = None,
    checkpoint_path: Optional[str | Path] = None,
    token_usage_path: Optional[str | Path] = None,
    resume_path: Optional[str | Path] = None,
    best_policies_dir: Optional[str | Path] = None,
    split_seed: Optional[int] = None,
) -> tuple[str, float, dict]:
    """
    Run the evolutionary loop.

    Fitness (survival): MSE on train_data.
    Best individual: lowest eval MSE.
    Final report: test MSE for best individual.

    Returns:
        (best_code, best_mse_eval, metrics_dict)
    """
    from .config import DEFAULT_CHECKPOINT_FILE, DEFAULT_TOKEN_USAGE_FILE

    if not train_data:
        raise ValueError("Empty train data")

    state_length = len(train_data[0]["state"])

    # Resolve model name
    if model_name is None:
        models = MODEL_TIERS.get(model_tier, MODEL_TIERS["cheap"])
        model_name = models[0]

    run_id = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    token_calls: list[dict] = []

    config_snapshot = {
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "test_size": len(test_data),
        "split_seed": split_seed,
        "population_size": population_size,
        "survivors_per_gen": survivors_per_gen,
        "mutations_per_survivor": mutations_per_survivor,
        "max_generations": max_generations,
        "model_name": model_name,
        "model_tier": model_tier,
    }

    def _checkpoint_and_exit(
        gen: int,
        pop: list[tuple[str, float]],
        best_c: str,
        best_m_eval: float,
        best_gen: int,
        hist: list[dict],
        msg: str,
        best_m_train: float | None = None,
    ) -> None:
        cpath = Path(checkpoint_path or DEFAULT_CHECKPOINT_FILE)
        tpath = Path(token_usage_path or DEFAULT_TOKEN_USAGE_FILE)
        _save_checkpoint(cpath, gen, pop, best_c, best_m_eval, best_gen, hist, config_snapshot, best_m_train)
        _save_token_usage(tpath, run_id, model_name, token_calls)
        print(msg)
        print(f"Checkpoint saved to {cpath}. Resume with: --resume {cpath}")
        raise SystemExit(1)

    population: list[tuple[str, float]] = []  # (code, train_mse)
    best_code = ""
    best_mse_eval = float("inf")
    best_mse_train = float("inf")
    best_generation = 0
    generations_without_improvement = 0
    metrics_history: list[dict] = []
    start_gen = 0

    def _best_by_eval(pop: list[tuple[str, float]]) -> tuple[str, float, float]:
        """Return (code, train_mse, eval_mse) for individual with lowest eval MSE."""
        best_c, best_tm, best_em = "", float("inf"), float("inf")
        for code, train_mse in pop:
            eval_mse, ok = evaluate(code, eval_data)
            if ok and eval_mse < best_em:
                best_c, best_tm, best_em = code, train_mse, eval_mse
        return best_c, best_tm, best_em

    if resume_path:
        ckpt = _load_checkpoint(Path(resume_path))
        population = [(r["code"], r["mse"]) for r in ckpt["population"]]
        best_code = ckpt["best_code"]
        best_mse_eval = ckpt.get("best_mse_eval", ckpt.get("best_mse", float("inf")))
        best_mse_train = ckpt.get("best_mse_train", best_mse_eval)
        best_generation = ckpt["best_generation"]
        metrics_history = ckpt["metrics_history"]
        start_gen = ckpt["generation"] + 1
        cfg = ckpt.get("config_snapshot", {})
        if cfg.get("model_name"):
            model_name = cfg["model_name"]
        print(f"Resumed from generation {ckpt['generation']}. Continuing from gen {start_gen}.")
    else:
        # Initialize population via LLM
        try:
            initial, stats_list = generate_initial_population(
                population_size, model_name, state_length=state_length
            )
            for s in stats_list:
                token_calls.append({"call": "generate_initial", **s})
        except Exception as e:
            print(f"LLM failed to generate initial population: {e}")
            raise

        if not initial:
            raise RuntimeError("LLM failed to generate any initial candidates")

        for code in initial:
            mse, ok = evaluate(code, train_data)
            population.append((code, mse))

        population.sort(key=lambda x: x[1])
        best_code, best_mse_train, best_mse_eval = _best_by_eval(population)
        metrics_history = [
            {
                "generation": 0,
                "best_mse_train": float(best_mse_train),
                "best_mse_eval": float(best_mse_eval),
                "population_mse": [float(m) for _, m in population],
                "best_generation": 0,
            }
        ]
        print(f"Gen 0 (initial): best_mse_train={best_mse_train:.6f}, best_mse_eval={best_mse_eval:.6f}")

        if best_policies_dir and best_code:
            Path(best_policies_dir).mkdir(parents=True, exist_ok=True)
            (Path(best_policies_dir) / "gen_0.py").write_text(best_code, encoding="utf-8")

        if checkpoint_path:
            _save_checkpoint(
                Path(checkpoint_path),
                0,
                population,
                best_code,
                best_mse_eval,
                best_generation,
                metrics_history,
                config_snapshot,
            )
        if token_usage_path:
            _save_token_usage(Path(token_usage_path), run_id, model_name, token_calls)

    for gen in range(start_gen, max_generations + 1):
        survivors = population[:survivors_per_gen]
        new_individuals: list[tuple[str, float]] = []

        for code, mse in survivors:
            for _ in range(mutations_per_survivor):
                try:
                    mutated, stats = mutate(
                        code, mse, model_name, state_length=state_length
                    )
                    token_calls.append({"call": "mutate", **stats})
                    if mutated and mutated != code:
                        mse_new, ok = evaluate(mutated, train_data)
                        new_individuals.append((mutated, mse_new))
                except Exception as e:
                    _checkpoint_and_exit(
                        gen - 1 if gen > 0 else 0,
                        population,
                        best_code,
                        best_mse_eval,
                        best_generation,
                        metrics_history,
                        f"LLM mutate failed after retries: {e}",
                        best_mse_train,
                    )

        if fresh_injection_per_gen > 0:
            try:
                fresh, stats_list = generate_initial_population(
                    fresh_injection_per_gen, model_name, state_length=state_length
                )
                for s in stats_list:
                    token_calls.append({"call": "generate_initial", **s})
                for code in fresh:
                    mse, ok = evaluate(code, train_data)
                    new_individuals.append((code, mse))
            except Exception as e:
                _checkpoint_and_exit(
                        gen - 1 if gen > 0 else 0,
                        population,
                        best_code,
                        best_mse_eval,
                        best_generation,
                        metrics_history,
                        f"LLM fresh injection failed after retries: {e}",
                        best_mse_train,
                )

        population = list(survivors) + new_individuals
        population.sort(key=lambda x: x[1])
        population = population[:population_size]

        curr_best_code, curr_best_train, curr_best_eval = _best_by_eval(population)

        if curr_best_eval < best_mse_eval:
            best_code = curr_best_code
            best_mse_eval = curr_best_eval
            best_mse_train = curr_best_train
            best_generation = gen
            generations_without_improvement = 0
        else:
            generations_without_improvement += 1

        mse_list = [mse for _, mse in population]
        metrics_history.append(
            {
                "generation": gen,
                "best_mse_train": float(best_mse_train),
                "best_mse_eval": float(best_mse_eval),
                "population_mse": [float(m) for m in mse_list],
                "best_generation": best_generation,
            }
        )

        print(f"Gen {gen}: best_mse_train={best_mse_train:.6f}, best_mse_eval={best_mse_eval:.6f}, pop_min={min(mse_list):.6f}, pop_max={max(mse_list):.6f}")

        if best_policies_dir and best_code:
            Path(best_policies_dir).mkdir(parents=True, exist_ok=True)
            (Path(best_policies_dir) / f"gen_{gen}.py").write_text(best_code, encoding="utf-8")

        if checkpoint_path:
            _save_checkpoint(
                Path(checkpoint_path),
                gen,
                population,
                best_code,
                best_mse_eval,
                best_generation,
                metrics_history,
                config_snapshot,
                best_mse_train,
            )
        if token_usage_path:
            _save_token_usage(Path(token_usage_path), run_id, model_name, token_calls)

        if generations_without_improvement >= early_stop_patience:
            print(f"Early stop: no improvement for {early_stop_patience} generations")
            break

    # Final test MSE for best individual
    best_mse_test, _ = evaluate(best_code, test_data) if test_data else (float("nan"), False)

    metrics = {
        "best_mse_train": float(best_mse_train),
        "best_mse_eval": float(best_mse_eval),
        "best_mse_test": float(best_mse_test),
        "train_size": len(train_data),
        "eval_size": len(eval_data),
        "test_size": len(test_data),
        "best_generation": best_generation,
        "total_generations": len(metrics_history),
        "history": metrics_history,
    }

    print(f"Final: best_mse_train={best_mse_train:.6f}, best_mse_eval={best_mse_eval:.6f}, best_mse_test={best_mse_test:.6f}")

    if output_path:
        Path(output_path).write_text(best_code, encoding="utf-8")
        print(f"Best policy saved to {output_path}")

    if metrics_path:
        Path(metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        print(f"Metrics saved to {metrics_path}")

    if token_usage_path:
        _save_token_usage(Path(token_usage_path), run_id, model_name, token_calls)

    return best_code, best_mse_eval, metrics
