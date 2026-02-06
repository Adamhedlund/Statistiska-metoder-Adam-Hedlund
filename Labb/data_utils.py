import numpy as np

data = np.genfromtxt(
    "../Data/housing.csv",
    delimiter=",",
    names=True,
    dtype=None,
    encoding="utf-8")

