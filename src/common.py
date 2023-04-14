# filename "common.py"
# Global methods: contains general methods used everywhere

import numpy as np
from scipy.constants import c, k

# Global dictionaries
node_labels = {'BS': 0, 'UE': 1, 'RIS': 2}

# The following are defined for graphic purpose only
node_color = {'BS': '#DC2516',  'UE': '#36F507', 'RIS': '#0F4EEA'}
node_mark = {'BS': 'o', 'UE': 'x', 'RIS': '^'}

# Custom distributions
def circ_uniform(n: int, r_outer: float, r_inner: float = 0, rng: np.random.RandomState = None):
    """Generate n points uniform distributed on an annular region. The outputs
    is given in polar coordinates.

    Parameters
    ----------
    n : int,
        number of points.
    r_outer : float,
        outer radius of the annular region.
    r_inner : float,
        inner radius of the annular region.

    Returns
    -------
    rho : np.ndarray,
        distance of each point from center of the annular region.
    phi : np.ndarray,
        azimuth angle of each point.
    """
    if rng is None:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * np.random.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * np.random.rand(n, 1)
    else:
        rho = np.sqrt((r_outer ** 2 - r_inner ** 2) * rng.rand(n, 1) + r_inner ** 2)
        phi = 2 * np.pi * rng.rand(n, 1)
    return rho, phi


# Physical noise
def thermal_noise(bandwidth, noise_figure=3, t0=293):
    """Compute the noise power [dBm] according to bandwidth and ambient temperature.

    :param bandwidth : float, receiver total bandwidth [Hz]
    :param noise_figure: float, noise figure of the receiver [dB]
    :param t0: float, ambient temperature [K]

    :return: power of the noise [dBm]
    """
    return watt2dbm(k * bandwidth * t0) + noise_figure  # [dBm]


# Utilities
def dbm2watt(dbm):
    """Simply converts dBm to Watt"""
    return 10 ** (dbm / 10 - 3)


def watt2dbm(watt):
    """Simply converts Watt to dBm"""
    with np.errstate(divide='ignore'):
        return 10 * np.log10(watt * 1e3)