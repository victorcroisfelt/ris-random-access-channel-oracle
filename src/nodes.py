#!/usr/bin/env python3
# filename "nodes.py"

import numpy as np
from scipy.constants import speed_of_light

import matplotlib.pyplot as plt


class Node:
    """Creates a communication entity.

    Arguments
    ---------
        n : int
            Number of nodes.

        pos : ndarray of shape (n, 3) or (3,) if n = 1
          Position of the node in rectangular coordinates.

        gain : float
            Antenna gain of the node.

        max_pow : float
         Max power available on transmission in linear scale.
    """

    def __init__(
            self,
            n: int,
            pos: np.ndarray,
            gain: float or np.ndarray = None,
            max_pow: float or np.ndarray = None,
    ):

        # Control on INPUT
        if pos.shape != (n, 3) and pos.shape != (3, ):
            raise ValueError(f'Illegal positioning: for Node, pos.shape must be ({n}, 3), instead it is {pos.shape}')

        # Set attributes
        self.n = n
        self.pos = pos
        self.gain = gain
        self.max_pow = max_pow


class BS(Node):
    """Base station.

    Arguments
    ---------
        pos : ndarray of shape (3,)
            Position of the BS in rectangular coordinates.

        gain : float
            BS antenna gain. Default is 5.00 dB.

        max_pow : float
            BS max power. Default is 30 dBm.
    """

    def __init__(
            self,
            n: int = None,
            pos: np.ndarray = None,
            gain: float = None,
            max_pow: float = None,
    ):
        if n is None:
            n = 1
        if gain is None:
            gain = 10**(5/10)
        if max_pow is None:
            max_pow = 100  # [mW]

        # Init parent class
        super().__init__(n, pos, gain, max_pow)

        self.distance = np.linalg.norm(self.pos)
        self.angle = np.abs(np.arctan2(self.pos[0], self.pos[1]))

    def __repr__(self):
        return f'BS-{self.n}'


class UE(Node):
    """User.

    Arguments
    ---------
        n : int
            Number of UEs.

        pos : ndarray of shape (n, 3)
            Position of the UEs in rectangular coordinates.

        gain : float
            BS antenna gain. Default is 5.00 dB.

        max_pow : float
            BS max power. Default is 30 dBm.
    """

    def __init__(
            self,
            n: int,
            pos: np.ndarray,
            gain: float = None,
            max_pow: float = None,
    ):

        if gain is None:
            gain = 10**(5/10)
        if max_pow is None:
            max_pow = 10  # [mW]

        # Init parent class
        super().__init__(n, pos, gain, max_pow)

        self.distances = np.linalg.norm(self.pos, axis=-1)
        self.angles = np.abs(np.arctan2(self.pos[:, 0], self.pos[:, 1]))

    def __repr__(self):
        return f'UE-{self.n}'


class RIS(Node):
    """Reflective Intelligent Surface.

    Arguments
    ---------
        pos : ndarray of shape (3,)
            Position of the RIS in rectangular coordinates.

        num_els_ver : int
            Number of elements along z-axis.

        num_els_hor : int
            Number of elements along x-axis.

        wavelength : float
            Wavelength in meters. Default: assume carrier frequency of 3 GHz.

        size_el : float
            Size of each element. Default: wavelength/4

        num_configs : int
            Number of configurations.
    """

    def __init__(
            self,
            n: int = None,
            pos: np.ndarray = None,
            num_els_ver: int = None,
            num_els_hor: int = None,
            wavelength: float = None,
            size_el: float = None,
    ):

        # Default values
        if n is None:
            n = 1
        if pos is None:
            pos = np.array([0, 0, 0])
        if num_els_ver is None:
            num_els_ver = 10
        if num_els_hor is None:
            num_els_hor = 10
        if wavelength is None:
            carrier_frequency = 3e9
            wavelength = speed_of_light / carrier_frequency
        if size_el is None:
            size_el = wavelength/2

        # Initialize the parent, considering that the antenna gain of the ris is 0.0,
        # max_pow and noise_power are -np.inf,
        # the number of antenna is the number or RIS elements
        super().__init__(n, pos, 0.0, -np.inf)
        # In this way every ris instantiated is equal to the others

        # Instance general parameters
        self.num_els_ver = num_els_ver  # vertical number of elements
        self.num_els_hor = num_els_hor  # horizontal number of elements
        self.num_els = num_els_ver * num_els_hor  # total number of elements
        self.size_el = size_el

        # Compute RIS sizes
        self.size_z = num_els_ver * self.size_el  # vertical size [m]
        self.size_x = num_els_hor * self.size_el  # horizontal size [m]
        self.area = self.size_z * self.size_x   # area [m^2]

    def __repr__(self):
        return f'RIS-{self.n}'
