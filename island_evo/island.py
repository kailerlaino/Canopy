"""Island dataclass and single-island evolution step."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .evaluator import FAILURE_MSE, score
from .mutator import mutate
from .novelty import NoveltyFilter


@dataclass
class Island:
    island_id: int
    incumbent_code: str
    incumbent_mse: float
    stale_count: int = 0
    generation_born: int = 0


@dataclass
class IslandStepResult:
    island: Island
    improved: bool
    candidate_mse: float        # FAILURE_MSE if rejected or parse-failed
    novelty_rejected: bool
    llm_failed: bool            # True if LLM call itself raised
    token_stats: dict = field(default_factory=dict)


def step_island(
    island: Island,
    model_name: str,
    state_length: int,
    train_data: list,
    novelty_filter: NoveltyFilter,
    patience: int,
) -> IslandStepResult:
    """Advance one island by one generation.

    Steps:
      1. Mutate incumbent via LLM.
      2. Check novelty — if too similar to archive, discard without scoring.
      3. Score the candidate.
      4. Greedy selection: replace incumbent only if candidate MSE is lower.
      5. Register candidate in the novelty archive (whether it improved or not).

    Returns an IslandStepResult describing what happened.
    """
    # -- LLM mutation ----------------------------------------------------
    try:
        candidate_code, stats = mutate(
            island.incumbent_code, island.incumbent_mse, model_name, state_length
        )
    except RuntimeError:
        # LLM call exhausted retries — count as stale
        stale = island.stale_count + 1
        return IslandStepResult(
            island=Island(
                island_id=island.island_id,
                incumbent_code=island.incumbent_code,
                incumbent_mse=island.incumbent_mse,
                stale_count=stale,
                generation_born=island.generation_born,
            ),
            improved=False,
            candidate_mse=FAILURE_MSE,
            novelty_rejected=False,
            llm_failed=True,
        )

    # -- Parse failure ---------------------------------------------------
    if candidate_code is None:
        stale = island.stale_count + 1
        return IslandStepResult(
            island=Island(
                island_id=island.island_id,
                incumbent_code=island.incumbent_code,
                incumbent_mse=island.incumbent_mse,
                stale_count=stale,
                generation_born=island.generation_born,
            ),
            improved=False,
            candidate_mse=FAILURE_MSE,
            novelty_rejected=False,
            llm_failed=False,
            token_stats=stats,
        )

    # -- Novelty gate ----------------------------------------------------
    if not novelty_filter.is_novel(candidate_code):
        stale = island.stale_count + 1
        return IslandStepResult(
            island=Island(
                island_id=island.island_id,
                incumbent_code=island.incumbent_code,
                incumbent_mse=island.incumbent_mse,
                stale_count=stale,
                generation_born=island.generation_born,
            ),
            improved=False,
            candidate_mse=FAILURE_MSE,
            novelty_rejected=True,
            llm_failed=False,
            token_stats=stats,
        )

    # -- Evaluate --------------------------------------------------------
    candidate_mse = score(candidate_code, train_data)
    novelty_filter.register(candidate_code)

    improved = candidate_mse < island.incumbent_mse
    if improved:
        new_island = Island(
            island_id=island.island_id,
            incumbent_code=candidate_code,
            incumbent_mse=candidate_mse,
            stale_count=0,
            generation_born=island.generation_born,
        )
    else:
        new_island = Island(
            island_id=island.island_id,
            incumbent_code=island.incumbent_code,
            incumbent_mse=island.incumbent_mse,
            stale_count=island.stale_count + 1,
            generation_born=island.generation_born,
        )

    return IslandStepResult(
        island=new_island,
        improved=improved,
        candidate_mse=candidate_mse,
        novelty_rejected=False,
        llm_failed=False,
        token_stats=stats,
    )
