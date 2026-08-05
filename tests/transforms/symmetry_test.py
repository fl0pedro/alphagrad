import pytest as _pytest_quarantine

_pytest_quarantine.skip(
    "QUARANTINED 2026-08-05: ModuleNotFoundError: No module named 'graphax.transforms' "
    "The code under test no longer exists; kept for provenance so the suite "
    "can serve as a green/red gate. Delete or restore deliberately.",
    allow_module_level=True,
)

import jax
import jax.random as jrand

from graphax.transforms.symmetry import swap_rows, swap_cols, swap_intermediates
from graphax.examples import make_Helmholtz

edges, info = make_Helmholtz()
# print(swap_rows(2, 3, edges))
# print(swap_cols(2, 3, edges))
print(swap_intermediates(4, 5, edges, info))

