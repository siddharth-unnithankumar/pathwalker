import numpy as np
import pytest

from pathwalker.movement_mechanisms import Energy
from pathwalker.utils import fmean, window


@pytest.fixture
def surf() -> np.ndarray:
    """Resistance surface."""
    return np.array(((1, 2), (3, 4)))


@pytest.fixture
def energy_mm(surf) -> Energy:
    return Energy(
        ij=(0, 0),
        surf=surf,
        corr=0.0,
        scale=1,
        fm=fmean,
        dest=(1, 1),
        deg=0.0,
        steps=10,
        energy=1000,
    )


# weirdly, getting different failures when running the same test command
# are the fixtures being saved between runs?
def test_Energy(energy_mm):
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
