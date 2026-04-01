"""Migration policy for the island-based evolutionary system.

Trigger: every `migration_interval` generations.

Selection: tournament — the two islands with the lowest (best) incumbent MSE
are the donors.  Each donor's incumbent is dispatched to every other island
in ring order (island i receives from the donor assigned to slot i % 2).

Acceptance: a migrant replaces the recipient's incumbent only if
  migrant_mse < recipient_mse  (fitness-gated, greedy).

Novelty: migrants that survive the fitness gate are also checked against the
shared NoveltyFilter archive before being registered, so that identical code
arriving on multiple islands is not re-registered redundantly.
"""

from __future__ import annotations

from .island import Island
from .novelty import NoveltyFilter


def maybe_migrate(
    islands: list[Island],
    generation: int,
    migration_interval: int,
    novelty_filter: NoveltyFilter,
) -> list[Island]:
    """Conditionally run one migration round and return the updated island list.

    Does nothing and returns the list unchanged if the trigger condition is
    not met or there are fewer than two islands.
    """
    if generation == 0:
        return islands
    if len(islands) < 2:
        return islands
    if generation % migration_interval != 0:
        return islands

    return _migrate(islands, novelty_filter)


def _migrate(
    islands: list[Island],
    novelty_filter: NoveltyFilter,
) -> list[Island]:
    """Perform one migration round.

    1. Select two donors: the two distinct islands with the best MSE.
    2. For each non-donor island (ring position i):
         donor = donors[i % 2]
         if donor.incumbent_mse < recipient.incumbent_mse:
             replace recipient's incumbent with donor's code/mse.
    3. Register each migrant code in the novelty archive (once, deduped).
    """
    # Sort by MSE ascending; pick top-2 distinct islands as donors
    ranked = sorted(islands, key=lambda isl: isl.incumbent_mse)
    donors = ranked[:2]
    donor_ids = {d.island_id for d in donors}

    registered_codes: set[int] = set()  # id() of already-registered code strs

    def _register_once(code: str) -> None:
        key = id(code)
        if key not in registered_codes:
            novelty_filter.register(code)
            registered_codes.add(key)

    updated: list[Island] = []
    ring_index = 0  # counts only non-donor recipients
    for isl in islands:
        if isl.island_id in donor_ids:
            updated.append(isl)
            continue

        donor = donors[ring_index % len(donors)]
        ring_index += 1

        if donor.incumbent_mse < isl.incumbent_mse:
            _register_once(donor.incumbent_code)
            updated.append(
                Island(
                    island_id=isl.island_id,
                    incumbent_code=donor.incumbent_code,
                    incumbent_mse=donor.incumbent_mse,
                    stale_count=0,          # fresh start after receiving migrant
                    generation_born=isl.generation_born,
                )
            )
        else:
            updated.append(isl)

    return updated
