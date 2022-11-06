# authors: siddharth unnithan kumar & samuel alan cushman

import numpy as np
import matplotlib.pyplot as plt
import time

T = {
    "load settings": 0,
    "compute runs": 0,
    "output results": 0,
    "output figures": 0,
}

# -------------MECHANISMS-------------------------

# -AUXILIARY FUNCTIONS-

# COORDINATE FUNCTIONS
# here, ij is the coordinates (i,j), so ij[0]=i and ij[1]=j
# note that we now have 9 possible movement tiles instead of 8, with the option of staying in the same tile

# enhanced adjacency function
def A(ij, i0, j0):
    return ij[0] + i0, ij[1] + j0


# list of 9 adjacent coordinates, from left to right and then down (increasing j then increasing i)
A0 = []
for i0 in np.arange(-1, 2):
    for j0 in np.arange(-1, 2):
        A0.append((i0, j0))

# window coordinates, from left to right and then down (increasing j then increasing i)
def window(scale):
    w = []
    for i0 in np.arange(-scale, scale + 1):
        for j0 in np.arange(-scale, scale + 1):
            w.append((i0, j0))
    return w


# FOCAL FUNCTIONS
# here, a n-by-n window is given by a scale of (n-1)/2


def fmean(ij, surf, scale):
    w = window(scale)
    Aw = [A(ij, i0, j0) for i0, j0 in w]
    Sw = np.array([surf[x] for x in Aw])
    return np.mean(Sw)


def fmax(ij, surf, scale):
    w = window(scale)
    Aw = [A(ij, i0, j0) for i0, j0 in w]
    Sw = np.array([surf[x] for x in Aw])
    return Sw.max()


def fmin(ij, surf, scale):
    w = window(scale)
    Aw = [A(ij, i0, j0) for i0, j0 in w]
    Sw = np.array([surf[x] for x in Aw])
    return Sw.min()


focalfunctions = [fmean, fmax, fmin]


# INVERSE RESISTANCE VALUES FUNCTION
# returns the 9-vector of inverse resistance values, with a sqrt(2) weighting
# if normalised, this gives the movement probabilities
# here, fm is the choice of focal measure, such as focal mean or max


def R(ij, surf, scale, fm):
    resistances = np.array(
        [fm(A(ij, A0[k][0], A0[k][1]), surf, scale) for k in range(9)]
    )
    weightedresistances = resistances * np.array(
        [np.sqrt(2), 1, np.sqrt(2), 1, 1, 1, np.sqrt(2), 1, np.sqrt(2)]
    )
    return 1 / weightedresistances


# DESTINATION FUNCTION
# given a source and destination on a surface
# returns one of the 8 movement directions in A0
# in movement, deg + corr must be at most 1


def D(source, target):
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


# -MOVEMENT MECHANISMS-

# E
def En(ij, energy, steps, surf, risk, corr, scale, fm, dest, deg):

    # initial conditions
    w = window(scale)
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
    w = window(scale)
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
    w = window(scale)
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
    w = window(scale)
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
    w = window(scale)
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
    w = window(scale)
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
    w = window(scale)
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

# -------------SETTINGS-------------------------

print("session_name = ")
session_name = input()

# load settings
start = time.time()
settings = dict(
    np.genfromtxt(
        "{}/settings.txt".format(session_name), dtype=None, encoding=None
    )
)

# set the parameters
Pmechanism = Mech[int(settings["mechanism"]) - 1]
Pscale = int(settings["scale"]) - 1
Pfunc = focalfunctions[int(settings["scaling_function"]) - 1]
Pcorrelation = float(settings["correlation"])
Pdeg = float(settings["destination_bias"])
Psteps = int(settings["steps"])
Penergy = int(settings["energy"])

# load the resistance surface data
Psurface = np.genfromtxt(
    "{}/{}".format(session_name, settings["resistance_surface"]),
    dtype=float,
    skip_header=6,
    encoding=None,
)
info = np.genfromtxt(
    "{}/{}".format(session_name, settings["resistance_surface"]),
    dtype=None,
    max_rows=6,
    encoding=None,
)
info = [list(info[i])[1] for i in range(6)]
nodataindex = np.where(Psurface == info[5])

# convert the nodata value to np.inf and add barrier
th = Pscale + 2
Psurface[Psurface == info[5]] = np.inf
Rsurface = np.inf * np.ones(
    (Psurface.shape[0] + 2 * th, Psurface.shape[1] + 2 * th)
)
Rsurface[th:-th, th:-th] = Psurface

# rescale the resistance surface to between 1 and 100
Rsurface = Rsurface - Rsurface.min()
Rsurface = Rsurface * (99 / Rsurface[Rsurface < np.inf].max())
Rsurface = Rsurface + 1

# similarly, configure the risk surface
if settings["risk_surface"] == "0":
    risksurface = Rsurface / 1000
else:
    rstemp = np.genfromtxt(
        "{}/{}".format(session_name, settings["risk_surface"]),
        dtype=float,
        skip_header=6,
        encoding=None,
    )
    rstemp[rstemp == info[5]] = np.inf
    risksurface = np.inf * np.ones(
        (rstemp.shape[0] + 2 * th, rstemp.shape[1] + 2 * th)
    )
    risksurface[th:-th, th:-th] = rstemp

# load source xy coordinates
source_xy = np.genfromtxt(
    "{}/{}".format(session_name, settings["source_points"]), dtype=float
)

# transform into grid xy coordinates, then into ij entries
temp = th + (source_xy - np.array([info[2], info[3]])) / info[4]
IJ = np.zeros(source_xy.shape)
IJ[:, 0] = Rsurface.shape[0] - (temp[:, 1] + 0.5)
IJ[:, 1] = temp[:, 0] - 0.5
IJ = np.round(IJ).astype(int)

# similarly, configure the destination points
if settings["source_destination_pairing"] == "0":
    Pdeg = 0
    IJdest = np.array((int(Rsurface.shape[0] / 2), int(Rsurface.shape[1] / 2)))
else:
    dest_xy = np.genfromtxt(
        "{}/{}".format(session_name, settings["destination_points"]),
        dtype=float,
    )
    temp = th + (dest_xy - np.array([info[2], info[3]])) / info[4]
    IJdest = np.zeros(dest_xy.shape)
    IJdest[:, 0] = Rsurface.shape[0] - (temp[:, 1] + 0.5)
    IJdest[:, 1] = temp[:, 0] - 0.5
    IJdest = np.round(IJdest).astype(int)

T["load settings"] = time.time() - start

# ------------------RUNS------------------------

# perform the runs
start = time.time()
ij_paths, add_info, labels = [], [], []

if int(settings["source_destination_pairing"]) == 0:

    for g2 in range(IJ.shape[0]):
        for g1 in range(int(settings["runs"])):
            temp = Pmechanism(
                (IJ[g2, 0], IJ[g2, 1]),
                Penergy,
                Psteps,
                Rsurface,
                risksurface,
                Pcorrelation,
                Pscale,
                Pfunc,
                (IJdest[0], IJdest[1]),
                Pdeg,
            )
            ij_paths.append(np.transpose(np.array(temp[0:2])))
            add_info.append(temp[2])
            labels.append("path_{}_{}_{}".format(g2 + 1, 0, g1 + 1))

if int(settings["source_destination_pairing"]) == 1:

    for g3 in range(IJ.shape[0]):
        for g2 in range(IJdest.shape[0]):
            for g1 in range(int(settings["runs"])):
                temp = Pmechanism(
                    (IJ[g3, 0], IJ[g3, 1]),
                    Penergy,
                    Psteps,
                    Rsurface,
                    risksurface,
                    Pcorrelation,
                    Pscale,
                    Pfunc,
                    (IJdest[g2, 0], IJdest[g2, 1]),
                    Pdeg,
                )
                ij_paths.append(np.transpose(np.array(temp[0:2])))
                add_info.append(temp[2])
                labels.append("path_{}_{}_{}".format(g3 + 1, g2 + 1, g1 + 1))

if int(settings["source_destination_pairing"]) == 2:

    for g2 in range(IJ.shape[0]):
        for g1 in range(int(settings["runs"])):
            temp = Pmechanism(
                (IJ[g2, 0], IJ[g2, 1]),
                Penergy,
                Psteps,
                Rsurface,
                risksurface,
                Pcorrelation,
                Pscale,
                Pfunc,
                (IJdest[g2, 0], IJdest[g2, 1]),
                Pdeg,
            )
            ij_paths.append(np.transpose(np.array(temp[0:2])))
            add_info.append(temp[2])
            labels.append("path_{}_{}_{}".format(g2 + 1, g2 + 1, g1 + 1))

if int(settings["source_destination_pairing"]) == 3:

    for g3 in range(IJ.shape[0]):

        x = np.arange(IJ.shape[0])
        x = np.delete(x, g3)

        for g2 in x:
            for g1 in range(int(settings["runs"])):
                temp = Pmechanism(
                    (IJ[g3, 0], IJ[g3, 1]),
                    Penergy,
                    Psteps,
                    Rsurface,
                    risksurface,
                    Pcorrelation,
                    Pscale,
                    Pfunc,
                    (IJdest[g2, 0], IJdest[g2, 1]),
                    Pdeg,
                )
                ij_paths.append(np.transpose(np.array(temp[0:2])))
                add_info.append(temp[2])
                labels.append("path_{}_{}_{}".format(g3 + 1, g2 + 1, g1 + 1))

T["compute runs"] = time.time() - start


# create the output directory
import csv
import os
from itertools import chain, zip_longest

os.mkdir("{}_results".format(session_name))


# convert ij-paths to xy-paths, then write to csv file
start = time.time()

if settings["output"] in ["1", "12", "13", "123"]:

    xy_paths = []
    for k in range(len(ij_paths)):
        temp = np.zeros(ij_paths[k].shape)
        temp[:, 0] = ij_paths[k][:, 1] + 0.5
        temp[:, 1] = Rsurface.shape[0] - (ij_paths[k][:, 0] + 0.5)
        temp = (temp - th) * info[4] + np.array([info[2], info[3]])
        xy_paths.append(temp)

    pathlabels = []
    for i in range(len(labels)):
        pathlabels.append("{}_x".format(labels[i]))
        pathlabels.append("{}_y".format(labels[i]))

    xy_write = list(
        chain(
            *[np.transpose(xy_paths[k]).tolist() for k in range(len(ij_paths))]
        )
    )
    for i in range(len(xy_write)):
        xy_write[i].insert(0, pathlabels[i])

    export_data = zip_longest(*xy_write, fillvalue="")
    with open(
        "{}_results/{}_paths.csv".format(session_name, session_name),
        "w",
        encoding="ISO-8859-1",
        newline="",
    ) as myfile:
        wr = csv.writer(myfile)
        wr.writerows(export_data)
    myfile.close()


# write additional info to csv file
if settings["output"] in ["2", "12", "23", "123"]:

    for i in range(len(add_info)):
        add_info[i].insert(0, labels[i])

    export_data = zip_longest(*add_info, fillvalue="")
    with open(
        "{}_results/{}_addinfo.csv".format(session_name, session_name),
        "w",
        encoding="ISO-8859-1",
        newline="",
    ) as myfile:
        wr = csv.writer(myfile)
        wr.writerows(export_data)
    myfile.close()


# create density surface, impute nodata values, write to csv file
if settings["output"] in ["3", "13", "23", "123"]:

    pdsurface = np.zeros(Rsurface.shape)

    for i in range(len(ij_paths)):
        for j in range(ij_paths[i].shape[0]):
            s = (ij_paths[i][j, 0], ij_paths[i][j, 1])
            pdsurface[s] += 1

pdsurfaceout = pdsurface.copy()
pdsurfaceout[nodataindex[0] + th, nodataindex[1] + th] = info[5]

x = open("{}_results/{}_pdsurface.csv".format(session_name, session_name), "w")
x.write(
    "ncols         {}\nnrows         {}\nxllcorner     {}\nyllcorner     {}\ncellsize      {}\nNODATA_value  {}\n".format(
        info[0], info[1], info[2], info[3], info[4], info[5]
    )
)
x.close()
with open(
    "{}_results/{}_pdsurface.csv".format(session_name, session_name),
    "a",
    newline="",
) as file:
    mywriter = csv.writer(file, delimiter=" ")
    mywriter.writerows(pdsurfaceout[th:-th, th:-th])

T["output results"] = time.time() - start


# create the figures
start = time.time()
if settings["figures"] in ["1", "12"]:

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(Rsurface[th:-th, th:-th])
    (image,) = ax.plot(ij_paths[0][:, 1] - th, ij_paths[0][:, 0] - th)
    fig.savefig(
        "{}_results/{}_{}.png".format(session_name, session_name, labels[0])
    )

    for i in np.arange(1, len(ij_paths)):
        image.set_data(ij_paths[i][:, 1] - th, ij_paths[i][:, 0] - th)
        fig.savefig(
            "{}_results/{}_{}.png".format(
                session_name, session_name, labels[i]
            )
        )

if settings["figures"] in ["2", "12"]:
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(pdsurface[th:-th, th:-th])
    fig.savefig(
        "{}_results/{}_pdsurface.png".format(session_name, session_name)
    )
    ax.imshow(np.log(pdsurface[th:-th, th:-th] + 1))
    fig.savefig(
        "{}_results/{}_logpdsurface.png".format(session_name, session_name)
    )

T["output figures"] = time.time() - start


# view the times
print("load settings: %f seconds" % T["load settings"])
print("compute runs: %f seconds" % T["compute runs"])
print("output results: %f seconds" % T["output results"])
print("output figures: %f seconds" % T["output figures"])
