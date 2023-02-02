import numpy as np
import pytest
from typing import List, Tuple

from pathwalker.movement_mechanisms import Energy
from pathwalker.utils import fmean, window


@pytest.fixture
def surf() -> np.ndarray:
    """Resistance surface.

    This surface is a 2D square matrix of ones padded around with a grid of
    high-resistance points that should not be possible to cross.
    """
    return np.array(
        (
            (10000, 10000, 10000, 10000, 10000),
            (10000, 1, 1, 1, 10000),
            (10000, 1, 1, 1, 10000),
            (10000, 1, 1, 1, 10000),
            (10000, 10000, 10000, 10000, 10000),
        )
    )


@pytest.fixture
def energy_mm(surf) -> Energy:
    """An instance of the energy movement mechanism."""
    return Energy(
        ij=(0, 0),
        surf=surf,
        corr=0.0,
        scale=1,
        fm=fmean,
        dest=(1, 1),
        deg=0.0,
        steps=100,
        energy=1000,
    )


@pytest.fixture
def barrier_points() -> List[Tuple[int, int]]:
    """
    A list of coordinates that the path should never cross.

    In this case, it is the outer barrier of high-resistance points.
    """
    return [coord for coord in window(2) if coord not in window(1)]


def test_Energy(energy_mm, barrier_points):
    """
    This test checks that the energy movement mechanism is correctly
    initialised. It then performs a sequence of timesteps until the conditions
    are exhausted, and checks that the walker never passes into a barrier
    region.
    """
    # test that required parameters are set to initial values
    assert energy_mm.d == []
    assert energy_mm.t == 0
    assert energy_mm.z1 == 0
    # tets that the conditions for continuing are true before the first step
    assert energy_mm.continue_conditions()
    # perform the first step and then check params have been updated
    energy_mm.first_step()
    assert len(energy_mm.d) == 1
    assert energy_mm.t == 1
    # perform steps until condition fails
    while energy_mm.continue_conditions():
        last_timestep = energy_mm.t
        energy_mm.general_step()
        assert energy_mm.t == last_timestep + 1
        # check that the path doesn't cross any predefined barrier points
        assert (
            energy_mm.ij not in barrier_points
        ), f"Landed on barrier point {energy_mm.ij} on timestep {energy_mm.t}!"
