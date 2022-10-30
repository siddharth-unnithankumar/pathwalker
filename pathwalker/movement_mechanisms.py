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
        surf,
        corr,
        scale: int,
        fm: Callable,
        dest: Tuple[int, int],
        deg: int,
    ) -> None:
        self.surf = surf
        self.corr = corr
        self.scale = scale
        self.fm = fm
        self.dest = dest
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

    def run_mechanism(self) -> List[List[int], List[int], List[Any]]:
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
        surf,
        corr,
        scale: int,
        fm: Callable,
        dest: Tuple[int, int],
        deg: int,
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


# E
def En(
    ij: Tuple[int, int], energy, steps, surf, risk, corr, scale, fm, dest, deg
):

    # initial conditions
    d = []
    z1 = 0
    t = 0
    I = [ij[0]]
    J = [ij[1]]

    # first step
    b = np.ones(9)
    b = (1 - deg) * b / np.sum(b)
    b = np.concatenate((b, deg * np.array([1])))
    n = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0]], p=b)
    ij = A(ij, A0[n][0], A0[n][1])
    t += 1
    z1 += fm(ij, surf, scale)
    I.append(ij[0])
    J.append(ij[1])
    d.append(n)

    # subsequent steps
    while t < steps and z1 < energy and fm(ij, surf, scale) < np.inf:
        b = np.ones(9)
        b = (1 - deg - corr) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1]), corr * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0], d[-1]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        z1 += fm(ij, surf, scale)
        I.append(ij[0])
        J.append(ij[1])
        d.append(n)
    return [I, J, [t, z1]]


# A
def Att(ij, energy, steps, surf, risk, corr, scale, fm, dest, deg):

    # initial conditions
    d = []
    t = 0
    I = [ij[0]]
    J = [ij[1]]

    # first step
    b = R(ij, surf, scale, fm)
    if np.sum(b) > 0:
        b = (1 - deg) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        I.append(ij[0])
        J.append(ij[1])
        d.append(n)

    # subsequent steps
    while t < steps and fm(ij, surf, scale) < np.inf:
        b = R(ij, surf, scale, fm)
        b = (1 - deg - corr) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1]), corr * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0], d[-1]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        I.append(ij[0])
        J.append(ij[1])
        d.append(n)
    return [I, J, [t]]


# R
def Ri(ij, energy, steps, surf, risk, corr, scale, fm, dest, deg):

    # initial conditions
    d = []
    r = 0
    z = []
    t = 0
    I = [ij[0]]
    J = [ij[1]]

    # first step
    b = np.ones(9)
    b = (1 - deg) * b / np.sum(b)
    b = np.concatenate((b, deg * np.array([1])))
    n = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0]], p=b)
    ij = A(ij, A0[n][0], A0[n][1])
    t += 1

    if fm(ij, risk, scale) < 1:
        q = fm(ij, risk, scale)
        r = np.random.choice([0, 1], p=[1 - q, q])
    else:
        q = fm(ij, risk, scale)
        r = 1

    I.append(ij[0])
    J.append(ij[1])
    z.append(q)
    d.append(n)

    # subsequent steps
    while t < steps and r == 0 and fm(ij, surf, scale) < np.inf:
        b = np.ones(9)
        b = (1 - deg - corr) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1]), corr * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0], d[-1]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1

        if fm(ij, risk, scale) < 1:
            q = fm(ij, risk, scale)
            r = np.random.choice([0, 1], p=[1 - q, q])
        else:
            q = fm(ij, risk, scale)
            r = 1

        I.append(ij[0])
        J.append(ij[1])
        z.append(q)
        d.append(n)
    return [I, J, [t, sum(z)]]


# EA
def EA(ij, energy, steps, surf, risk, corr, scale, fm, dest, deg):

    # initial conditions
    d = []
    t = 0
    z1 = 0
    I = [ij[0]]
    J = [ij[1]]

    # first step
    b = R(ij, surf, scale, fm)
    if np.sum(b) > 0:
        b = (1 - deg) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        z1 += fm(ij, surf, scale)
        I.append(ij[0])
        J.append(ij[1])
        d.append(n)

    # subsequent steps
    while z1 < energy and t < steps and fm(ij, surf, scale) < np.inf:
        b = R(ij, surf, scale, fm)
        b = (1 - deg - corr) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1]), corr * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0], d[-1]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        z1 += fm(ij, surf, scale)
        I.append(ij[0])
        J.append(ij[1])
        d.append(n)
    return [I, J, [t, z1]]


# ER
def ER(ij, energy, steps, surf, risk, corr, scale, fm, dest, deg):

    # initial conditions
    d = []
    r = 0
    z = []
    z1 = 0
    t = 0
    I = [ij[0]]
    J = [ij[1]]

    # first step
    b = np.ones(9)
    b = (1 - deg) * b / np.sum(b)
    b = np.concatenate((b, deg * np.array([1])))
    n = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0]], p=b)
    ij = A(ij, A0[n][0], A0[n][1])
    t += 1
    z1 += fm(ij, surf, scale)

    if fm(ij, risk, scale) < 1:
        q = fm(ij, risk, scale)
        r = np.random.choice([0, 1], p=[1 - q, q])
    else:
        q = fm(ij, risk, scale)
        r = 1

    I.append(ij[0])
    J.append(ij[1])
    z.append(q)
    d.append(n)

    # subsequent steps
    while z1 < energy and r == 0 and fm(ij, surf, scale) < np.inf:
        b = np.ones(9)
        b = (1 - deg - corr) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1]), corr * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0], d[-1]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        z1 += fm(ij, surf, scale)

        if fm(ij, risk, scale) < 1:
            q = fm(ij, risk, scale)
            r = np.random.choice([0, 1], p=[1 - q, q])
        else:
            q = fm(ij, risk, scale)
            r = 1

        I.append(ij[0])
        J.append(ij[1])
        z.append(q)
        d.append(n)
    return [I, J, [t, z1, sum(z)]]


# AR
def AR(ij, energy, steps, surf, risk, corr, scale, fm, dest, deg):

    # initial conditions
    d = []
    t = 0
    r = 0
    z = []
    I = [ij[0]]
    J = [ij[1]]

    # first step
    b = R(ij, surf, scale, fm)
    if np.sum(b) > 0:
        b = (1 - deg) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1

        if fm(ij, risk, scale) < 1:
            q = fm(ij, risk, scale)
            r = np.random.choice([0, 1], p=[1 - q, q])
        else:
            q = fm(ij, risk, scale)
            r = 1

        I.append(ij[0])
        J.append(ij[1])
        z.append(q)
        d.append(n)

    # subsequent steps
    while t < steps and r == 0 and fm(ij, surf, scale) < np.inf:
        b = R(ij, surf, scale, fm)
        b = (1 - deg - corr) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1]), corr * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0], d[-1]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1

        if fm(ij, risk, scale) < 1:
            q = fm(ij, risk, scale)
            r = np.random.choice([0, 1], p=[1 - q, q])
        else:
            q = fm(ij, risk, scale)
            r = 1

        I.append(ij[0])
        J.append(ij[1])
        z.append(q)
        d.append(n)
    return [I, J, [t, sum(z)]]


# EAR
def EAR(ij, energy, steps, surf, risk, corr, scale, fm, dest, deg):

    # initial conditions
    d = []
    t = 0
    z1 = 0
    r = 0
    z = []
    I = [ij[0]]
    J = [ij[1]]

    # first step
    b = R(ij, surf, scale, fm)
    if np.sum(b) > 0:
        b = (1 - deg) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        z1 += fm(ij, surf, scale)

        if fm(ij, risk, scale) < 1:
            q = fm(ij, risk, scale)
            r = np.random.choice([0, 1], p=[1 - q, q])
        else:
            q = fm(ij, risk, scale)
            r = 1

        I.append(ij[0])
        J.append(ij[1])
        z.append(q)
        d.append(n)

    # subsequent steps
    while (
        z1 < energy and t < steps and r == 0 and fm(ij, surf, scale) < np.inf
    ):
        b = R(ij, surf, scale, fm)
        b = (1 - deg - corr) * b / np.sum(b)
        b = np.concatenate((b, deg * np.array([1]), corr * np.array([1])))
        n = np.random.choice(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, D(ij, dest)[0][0], d[-1]], p=b
        )
        ij = A(ij, A0[n][0], A0[n][1])
        t += 1
        z1 += fm(ij, surf, scale)

        if fm(ij, risk, scale) < 1:
            q = fm(ij, risk, scale)
            r = np.random.choice([0, 1], p=[1 - q, q])
        else:
            q = fm(ij, risk, scale)
            r = 1

        I.append(ij[0])
        J.append(ij[1])
        z.append(q)
        d.append(n)
    return [I, J, [t, z1, sum(z)]]


# parse
Mech = [En, Att, Ri, EA, ER, AR, EAR]