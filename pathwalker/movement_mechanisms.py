from abc import ABC, abstractmethod
from typing import Any, Callable, List, Tuple, Union

import numpy as np

from utils import A, A0, D, R


class MovementMechanism(ABC):
    """
    Superclass for all movement mechanisms.
    Implements the core functionality common to all such mechanisms.
    """

    def __init__(
        self,
        ij: Tuple[int, int],
        surf: np.ndarray,
        corr: float,
        scale: int,
        fm: Callable,
        dest: Tuple[int, int],
        deg: float,
    ) -> None:
        self.surf = surf
        if corr < 0 or corr > 1:
            raise ValueError("Autocorrelation degree must be between 0 and 1!")
        self.corr = corr
        self.scale = scale
        self.fm = fm
        self.dest = dest
        if deg < 0 or deg > 1:
            raise ValueError(
                "Destination bias degree must be between 0 and 1!"
            )
        self.deg = deg
        # set initial conditions
        self.ij = ij
        self.d = []
        self.t = 0
        self.I = [self.ij[0]]
        self.J = [self.ij[1]]

    @abstractmethod
    def continue_conditions(self) -> bool:
        """
        Abstract method for conditions upon which to continue the while loop.
        """
        pass

    @abstractmethod
    def first_step(self) -> None:
        """
        Abstract method for updating the parameters on the first time step.
        The time step `t` must be incremented by 1 in this step. Additionally,
        the lists `I`, `J`, `d` must all be updated in this step.
        TODO: is this something that can be asserted in the method?
        """
        pass

    @abstractmethod
    def general_step(self) -> None:
        """
        Abstract method for updating the parameters on a general time step.
        The time step `t` must be incremented by 1 in this step. Additionally,
        the lists `I`, `J`, `d` must all be updated in this step.
        TODO: is this something that can be asserted in the method?
        """
        pass

    @abstractmethod
    def get_final_data(self) -> List[Any]:
        """
        Abstract method for the additional data that is returned along with the
        lists of x and y coordinates when running this movement mechanism.
        Since this method is called after the final time step, note that the
        values taken by parameters in this method correspond to their final
        values at the end of the path.
        """
        pass

    def run_mechanism(self) -> List[List[Any]]:
        """
        Run the movement mechanism.
        """
        self.first_step()
        # may need to set an attribute (dictionary) for continue condition
        # arguments, and update this attribute within the general step
        while self.continue_conditions():
            self.general_step()
        final_data = self.get_final_data()
        return [self.I, self.J, final_data]

    def save_step(self, i: int, j: int, n: int) -> None:
        """
        Update the lists I, J and d.
        :param i: x-coordinate of new point
        :param j: y-coordinate of new point
        :param n: position of new point in relation to old point within window.
        """
        self.I.append(i)
        self.J.append(j)
        self.d.append(n)


class Energy(MovementMechanism):
    """
    Energy movement mechanism.
    """

    def __init__(
        self,
        ij: Tuple[int, int],
        surf: np.ndarray,
        corr: float,
        scale: int,
        fm: Callable,
        dest: Tuple[int, int],
        deg: float,
        steps: int,
        energy: Union[int, float],
    ):
        # initialise the base class
        super().__init__(ij, surf, corr, scale, fm, dest, deg)
        # additionally, set new parameters
        self.steps = steps
        self.energy = energy
        # set new initial condition
        self.z1 = 0

    def continue_conditions(self) -> bool:
        """
        Conditions on which to continue the path.
        """
        condition = (
            self.t < self.steps
            and self.z1 < self.energy
            and self.fm(self.ij, self.surf, self.scale) < np.inf
        )
        return condition

    # TODO: it seems that it's possible to factor out more common patterns
    # between the first step and general step into separate methods
    # it could also just be a function taking in self.deg as an argument
    def first_step(self) -> None:
        """
        First step for the energy movement mechanism.
        """
        b = np.ones(9)
        b = (1 - self.deg) * b / np.sum(b)
        b = np.concatenate((b, self.deg * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(self.ij, self.dest)[0][0]], p=b
        )
        # update rules
        self.ij = A(self.ij, A0[n][0], A0[n][1])
        self.t += 1
        self.z1 += self.fm(self.ij, self.surf, self.scale)
        # save new results
        self.save_step(i=self.ij[0], j=self.ij[1], n=n)

    def general_step(self) -> None:
        """
        General step for the energy movement mechanism.
        """
        b = np.ones(9)
        b = (1 - self.deg - self.corr) * b / np.sum(b)
        b = np.concatenate(
            (b, self.deg * np.array([1]), self.corr * np.array([1]))
        )
        n = np.random.choice(
            [
                0,
                1,
                2,
                3,
                4,
                5,
                6,
                7,
                8,
                D(self.ij, self.dest)[0][0],
                self.d[-1],
            ],
            p=b,
        )
        self.ij = A(self.ij, A0[n][0], A0[n][1])
        self.t += 1
        self.z1 += self.fm(self.ij, self.surf, self.scale)
        self.save_step(i=self.ij[0], j=self.ij[1], n=n)

    def get_final_data(self) -> List[Any]:
        """Get the final data used for this mechanism."""
        return [self.t, self.z1]
