import numpy as np
from scipy.integrate import cumtrapz
from matplotlib import pyplot as plt


def hamming(r, QUICK=True):
    """
    :param r: Vector of radii, usually equally spaced
    :return r:  Similar length vector, under Hamming distribution, used for apodization and NAP
    """
    a0 = 0.53836   # Hamming constant: Hamming = a0+(1-a0)*cos(pi*r)
    if QUICK:
        # using an approximation found in 'find_direct_invF_for_hamming.py'. based on curve fitting
        p = np.array([0.0520186, 0.13813714, -0.03060542, -0.14245331, 1.81273629, 3.42201622, 0.31436716, -0.49470507, 1.89936756])
        invF = lambda x: p[0] * (x - 0.5) + p[1] * (x - 0.5) ** 2 + p[2] * (x - 0.5) ** 3 + p[3] * (x - 0.5) ** 4 + p[4] * (x - 0.5) ** 5 + p[5] * (x - 0.5) ** 6 + p[6] * np.arcsin(p[8] * (x - (1 / p[8]))) + p[7] + np.pi * p[6]
        r = invF(r)
    else:
        res = int(1e4)
        X = np.linspace(0, 1, res)
        n = np.size(r)

        F = lambda x: (0.5*a0*(x**2)+((1-a0)/(np.pi**2))*(np.pi*x*np.sin(np.pi*x)+np.cos(np.pi*x))-((1-a0)/(np.pi**2)))/(0.5*a0-2*((1-a0)/(np.pi**2)))
        # PD = cumtrapz((a0 + (1 - a0) * np.cos(np.pi * X)) * X, X)  # multiplied by X to have 2D distribution (Jacobian)
        # PD = PD/max(PD)
        PD = F(X)

        # r = np.linspace(1/n, 1, n)
        # r *= r
        r = np.tile(r, (PD.size, 1))
        PD = np.transpose(np.tile(PD, (n, 1)))
        ge = np.greater(r, PD)
        r = np.sum(ge, 0)/res

    return r


def uniform(r):
    return np.sqrt(r)
