"""Thin re-export of the Phase 1 evaluator interface.

Do not add logic here.  This file exists so island_evo imports never reach
into naive_baseline directly, and so Phase 3 has a single clean swap point
if the scoring contract ever changes.
"""

import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from naive_baseline.evaluator import score  # noqa: F401 — re-export
from evolutionary_mimic.fitness import FAILURE_MSE  # noqa: F401 — re-export
