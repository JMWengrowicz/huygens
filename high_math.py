import numpy as np


def angular_integration(A):

    y_size, x_size = np.shape(A)
    x_axis = np.linspace(-(x_size + 1) / 2, (x_size + 1) / 2, x_size)
    y_axis = np.linspace(-(y_size + 1) / 2, (y_size + 1) / 2, y_size)
    X, Y = np.meshgrid(x_axis, y_axis)
    R = np.round(np.sqrt(X**2 + Y**2))

    dist_vec, idx_vec, counts_vec = np.unique(R, return_inverse=True, return_counts=True)
    # dist_vec = unique(R)
    # vec1, bins = histc(R[:], dist_vec)
    bins = idx_vec
    vec1 = counts_vec

    A_vec = np.reshape(A, A.size)
    dist_int = np.bincount(bins, A_vec)
    dist_mean = dist_int/vec1
    std1 = np.bincount(bins, (A_vec-dist_mean[bins])**2)/vec1  # based on definition of std
    # std1 = accumarray(bins, A[:], [], np.std)

    output = np.array([dist_vec, dist_int, vec1, dist_mean]).transpose()
    PSFnorm = dist_mean / max(dist_mean)
    STDnorm = std1 / max(dist_mean)

    return output, PSFnorm, STDnorm


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return rho, phi


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y
