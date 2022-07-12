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

        num_els_z : int
            Number of elements along z-axis.

        num_els_x : int
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
            num_els_z: int = None,
            num_els_x: int = None,
            wavelength: float = None,
            size_el: float = None,
            num_configs: int = None,

    ):
        # Default values
        if n is None:
            n = 1
        if pos is None:
            pos = np.array([0, 0, 0])
        if num_els_z is None:
            num_els_z = 10
        if num_els_x is None:
            num_els_x = 10
        if wavelength is None:
            carrier_frequency = 3e9
            wavelength = speed_of_light / carrier_frequency
        if size_el is None:
            size_el = wavelength/2
        if num_configs is None:
            num_configs = 4

        # Initialize the parent, considering that the antenna gain of the ris is 0.0,
        # max_pow and noise_power are -np.inf,
        # the number of antenna is the number or RIS elements
        super().__init__(n, pos, 0.0, -np.inf)
        # In this way every ris instantiated is equal to the others

        # Instance variables
        self.num_els_z = num_els_z  # vertical number of elements
        self.num_els_x = num_els_x  # horizontal number of elements
        self.num_els = num_els_z * num_els_x  # total number of elements
        self.size_el = size_el
        self.num_configs = num_configs  # number of configurations

        # # Store index of elements considering total number
        # self.els_range = np.arange(self.num_els)

        # Compute RIS sizes
        self.size_z = num_els_z * self.size_el  # vertical size [m]
        self.size_x = num_els_x * self.size_el  # horizontal size [m]
        self.area = self.size_z * self.size_x   # area [m^2]

        # # Organizing elements over the RIS
        # self.id_els = self.indexing_els()
        # self.pos_els = self.positioning_els()

        # Configure RIS
        self.angular_resolution = None
        self.set_angular_resolution()

        self.configs = None
        self.set_configurations()

    def set_angular_resolution(self):
        """Set RIS angular resolution. The observation space is ever considered to be 0 to pi/2 (half-plane) given our
        system setup.

        Returns
        -------
        angular_resolution : float
            RIS angular resolution in radians given the number of configurations and uniform division of the observation
            space.

        Example
        -------
        For num_configs = 4, angular_resolution evaluates to pi/8.

        """
        self.angular_resolution = ((np.pi / 2) - 0) / self.num_configs

    def set_configurations(self):
        """Set configurations offered by the RIS.

        Returns
        -------
        set_configs : ndarray of shape (self.num_configs,)
            Discrete set of configurations containing all possible angles (theta_s) in radians in which the RIS can
            steer the incoming signal.

        Example
        -------
        For S = 4, angular resolution is pi/8. The set of configurations evaluates to:

                                 set_configs = [1/2, 3/2, 5/2, 7/2] * pi/8

        0 and pi/2 are not included. Note that the observation space is divided into 5 zones.
        """
        configs = np.arange(self.angular_resolution / 2, np.pi / 2, self.angular_resolution)

        assert len(configs) == self.num_configs, "Cardinality of configurations does not meet the number of configurations!"

        self.configs = configs

    def __repr__(self):
        return f'RIS-{self.n}'


    # def indexing_els(self):
    #     """Define an array of tuples where each entry represents the ID of an element.
    #
    #     Returns
    #     -------
    #     id_els : ndarray of tuples of shape (self.num_els)
    #         Each ndarray entry has a tuple (id_v, id_h), which indexes the elements arranged in a planar array. Vertical
    #         index is given as id_v, while horizontal index is id_h.
    #
    #     Example
    #     -------
    #     For a num_els_v = 3 x num_els_h = 3 RIS, the elements are indexed as follows:
    #
    #                                             (2,0) -- (2,1) -- (2,2)
    #                                             (1,0) -- (1,1) -- (1,2)
    #                                             (0,0) -- (0,1) -- (0,2),
    #
    #     the corresponding id_els should contain:
    #
    #                     id_els = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)].
    #
    #     While the respective enumeration of the elements is:
    #
    #                                                 6 -- 7 -- 8
    #                                                 3 -- 4 -- 5
    #                                                 0 -- 1 -- 2,
    #
    #     the enumeration is stored at:
    #
    #                                         self.els_range = np.arange(num_els).
    #
    #     Therefore, id_els and self.els_range are two different index methods for the elements. The former is used to
    #     characterize the geometrical features of each element, while the latter is used for storage purposes.
    #     """
    #     # Get vertical ids
    #     id_v = self.els_range // self.num_els_v
    #
    #     # Get horizontal ids
    #     id_h = np.mod(self.els_range, self.num_els_h)
    #
    #     # Get array of tuples with complete id
    #     id_els = [(id_v[el], id_h[el]) for el in self.els_range]
    #
    #     return id_els
    #
    # def positioning_els(self):
    #     """Compute position of each element in the planar array.
    #
    #     Returns
    #     -------
    #
    #     """
    #     # Compute offsets
    #     offset_x = (self.num_els_h - 1) * self.size_el / 2
    #     offset_z = (self.num_els_v - 1) * self.size_el / 2
    #
    #     # Prepare to store the 3D position vector of each element
    #     pos_els = np.zeros((self.num_els, 3))
    #
    #     # Go through all elements
    #     for el in self.els_range:
    #         pos_els[el, 0] = (self.id_els[el][1] * self.size_el) - offset_x
    #         pos_els[el, 2] = (self.id_els[el][0] * self.size_el) - offset_z
    #
    #     return pos_els
    #
    # def plot(self):
    #     """Plot RIS along with the index of each element.
    #
    #     Returns
    #     -------
    #     None.
    #
    #     """
    #     fig, ax = plt.subplots()
    #
    #     # Go through all elements
    #     for el in self.els_range:
    #         ax.plot(self.pos_els[el, 0], self.pos_els[el, 2], 'x', color='black')
    #         ax.text(self.pos_els[el, 0] - 0.003, self.pos_els[el, 2] - 0.0075, str(self.id_els[el]))
    #
    #     # Plot origin
    #     ax.plot(0, 0, '.', color='black')
    #
    #     ax.set_xlim([np.min(self.pos_els[:, 0]) - 0.05, np.max(self.pos_els[:, 0]) + 0.05])
    #     ax.set_ylim([np.min(self.pos_els[:, 2]) - 0.05, np.max(self.pos_els[:, 2]) + 0.05])
    #
    #     ax.set_xlabel("x [m]")
    #     ax.set_ylabel("z [m]")
    #
    #     plt.show()

# class RxNoise:
#     """Represent the noise value at the physical receiver
#     # TODO: match with ambient noise and noise figure
#     """
#
#     def __init__(self, linear=None, dB=None, dBm: np.ndarray = np.array([-92.5])):
#         if (linear is None) and (dB is None):
#             self.dBm = dBm
#             self.dB = dBm - 30
#             self.linear = 10 ** (self.dB / 10)
#         elif (linear is not None) and (dB is None):
#             self.linear = linear
#             if self.linear != 0:
#                 self.dB = 10 * np.log10(self.linear)
#                 self.dBm = 10 * np.log10(self.linear * 1e3)
#             else:
#                 self.dB = -np.inf
#                 self.dBm = -np.inf
#         else:
#             self.dB = dB
#             self.dBm = dB + 30
#             self.linear = 10 ** (self.dB / 10)
#
#     def __repr__(self):
#         return (f'noise({self.linear:.3e}, '
#                 f'dB={self.dB:.1f}, dBm={self.dBm:.1f})')
