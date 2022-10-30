"""Utilities module."""
import itertools
from typing import Callable, List, Tuple

import numpy as np


# COORDINATE FUNCTIONS
# here, ij is the coordinates (i,j), so ij[0]=i and ij[1]=j
# note that we now have 9 possible movement tiles instead of 8, with the option
# of staying in the same tile

# enhanced adjacency function
def A(ij: Tuple[int, int], i0: int, j0: int) -> Tuple[int, int]:
    """
    Shift a tuple ij by a specific horizontal and vertical distance.
    :param ij: input tuple of integers
    :param i0: horizontal integer distance
    :param j0: vertical integer distance
    :return: the tuple ij shifted by i0 horizontally and j0 vertically.
    """
    return ij[0] + i0, ij[1] + j0


# window coordinates, from left to right and then down (increasing j then
# increasing i)
def window(scale: int) -> List[Tuple[int, int]]:
    """
    Create a square window of a specific width `scale` around the origin.
    :param scale: an integer specifying the window size
    :return: a list of all coordinates in the window, centered around (0, 0).
    """
    return list(itertools.product(range(-scale, scale + 1), repeat=2))


# FOCAL FUNCTIONS
# here, a n-by-n window is given by a scale of (n-1)/2


def get_sw(ij: Tuple[int, int], surf, scale: int) -> np.ndarray:
    """
    TODO: figure out what this is!
    """
    w = window(scale)
    Aw = [A(ij, i0, j0) for i0, j0 in w]
    Sw = np.array([surf[x] for x in Aw])
    return Sw


def fmean(ij: Tuple[int, int], surf, scale: int) -> float:
    """
    Focal mean.
    """
    Sw = get_sw(ij=ij, surf=surf, scale=scale)
    return Sw.max()


def fmax(ij: Tuple[int, int], surf, scale: int) -> float:
    """
    Focal maximum.
    """
    Sw = get_sw(ij=ij, surf=surf, scale=scale)
    return Sw.max()


def fmin(ij: Tuple[int, int], surf, scale: int) -> float:
    """
    Focal minimum.
    """
    Sw = get_sw(ij=ij, surf=surf, scale=scale)
    return Sw.min()


def R(ij: Tuple[int, int], surf, scale: int, fm: Callable) -> np.ndarray:
    """
    Inverse resistance values function.
    Returns the 9-vector of inverse resistance values, with a sqrt(2) weighting
    When normalised, this gives the movement probabilities
    :param ij: the current coordinate
    :param surf: the resistance surface
    :param scale: the scaling
    :param fm: the choice of focal measure
    :return: the 9-dimensional vector of inverse resistance values
    """
    resistances = np.array(
        [fm(A(ij, A0[k][0], A0[k][1]), surf, scale) for k in range(9)]
    )
    weighted_resistances = resistances * np.array(
        [np.sqrt(2), 1, np.sqrt(2), 1, 1, 1, np.sqrt(2), 1, np.sqrt(2)]
    )
    return 1 / weighted_resistances


def D(source: Tuple[int, int], target: Tuple[int, int]):
    """
    Destination function.
    Given a source and destination on a surface, returns one of the 8 movement
    directions in A0.
    In movement, deg + corr must be at most 1.
    """
    if source == target:
        x = (np.array([4]),)
    else:
        v = np.array(target) - np.array(source)
        v = v / np.linalg.norm(v)
        d = np.array(
            [
                np.dot(v, np.array(z) / np.linalg.norm(np.array(z)))
                for z in [
                    (-1, -1),
                    (-1, 0),
                    (-1, 1),
                    (0, -1),
                    (0, 1),
                    (1, -1),
                    (1, 0),
                    (1, 1),
                ]
            ]
        )
        d = np.insert(d, 4, 0)
        x = np.where(d == d.max())
    return x


# fix the value A0 for later use
# it consists of the 9 possible directions in which to move on a given tile
# (with the origin corresponding to remaining on the same tile)
A0 = window(scale=1)

focalfunctions = [fmean, fmax, fmin]
