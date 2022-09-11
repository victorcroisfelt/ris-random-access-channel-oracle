import numpy as np

from scipy.constants import speed_of_light

import src.common as common
from src.nodes import UE, BS, RIS

#from collections import OrderedDict

import matplotlib.pyplot as plt
from matplotlib import rc

class Box:
    """Creates an environment defined by a box of UEs.

    Arguments:
        maximum_distance : float
            Maximum distance of the UEs with respect to the RIS.

        minimum_distance : float
            Minimum distance with respect to the RIS.

        carrier_frequency : float
            Central frequency in Hertz.

        bandwidth : float
            Bandwidth in Hertz.

    """

    def __init__(
            self,
            maximum_distance: float,
            minimum_distance: float,
            carrier_frequency: float = 3e9,
            bandwidth: float = 180e3,
            rng: np.random.RandomState = None
    ):

        # if (square_length < min_square_length) or (min_square_length <= 0) or (square_length <= 0):
        #     raise ValueError('Invalid definition of the box concerning its length.')

        # Physical attributes of the box
        self.maximum_distance = maximum_distance
        self.minimum_distance = minimum_distance

        # Bandwidth available
        self.fc = carrier_frequency
        self.wavelength = speed_of_light / carrier_frequency
        self.wavenumber = 2 * np.pi / self.wavelength
        self.bw = bandwidth

        # Random State generator
        self.rng = np.random.RandomState() if rng is None else rng

        # Nodes
        self.bs = None
        self.ue = None
        self.ris = None

    def place_bs(
            self,
            distance: float = None,
            zenith_angle_deg: float = None,
            gain: float = None,
            max_pow: float = None
    ):
        """Place a single BS in the environment. Following the paper, the BS is always located at the second quadrant of
         the coordinate system. If a new BS is set the old one is canceled.

        Parameters
        ----------
        distance : float
            Distance to the origin "0" of the coordinate system. Default: 25 meters.

        zenith_angle_deg : float
            Zenith angle in degrees. Default: 45 degrees

        gain : float
            BS antenna gain G_b.

        max_pow : float
           Maximum power available at the BS.
        """
        if distance is None:
            distance = 25
        if zenith_angle_deg is None:
            zenith_angle_deg = 45

        # Compute rectangular coordinates
        pos = distance * np.array([-np.sin(zenith_angle_deg), np.cos(zenith_angle_deg), 0])

        # Append BS
        self.bs = BS(1, pos, gain, max_pow)

    def place_ris(self,
                  pos: np.ndarray = None,
                  num_els_ver: int = None,
                  num_els_hor: int = None,
                  size_el: float = None,
                  # num_ce_configs: int = None,
                  # num_access_configs: int = None
                  ):
        """Place a single RIS in the environment. If a new RIS is set the old one is canceled.

        Parameters
        ----------

        pos : ndarray of shape (3,)
            Position of the RIS in rectangular coordinates.

        num_els_ver : int
            Number of elements along z-axis.

        num_els_hor : int
            Number of elements along x-axis.

        size_el : float
            Size of each element. Default: wavelength

        num_configs : int
            Number of configurations.
        """

        # Append RIS
        self.ris = RIS(
            pos=pos,
            num_els_ver=num_els_ver,
            num_els_hor=num_els_hor,
            wavelength=self.wavelength,
            size_el=size_el,
            # num_ce_configs=num_ce_configs,
            # num_access_configs=num_access_configs
        )

    def place_ue(
            self,
            n: int,
            gain: float = None,
            max_pow: float = None
    ):
        """Place a predefined number n of UEs in the box. If a new set of UE is set the old one is canceled.

        Parameters
        ----------

        n : int
            Number of UEs to be placed.

        gain : float
            UE antenna gain G_k.

        max_pow : float
           Maximum power available at each UE.
        """
        # # Control on input
        # if not isinstance(n, int) or (n <= 0):  # Cannot add a negative number of nodes
        #     raise ValueError('n must be int >= 0.')

        # Compute distances
        distances = np.sqrt(self.rng.rand(n, 1) * (self.maximum_distance**2 - self.minimum_distance**2) + self.minimum_distance**2)
        distances = np.squeeze(distances)

        # Compute angles
        angles = np.pi/2 * self.rng.rand(n, 1)
        angles = np.squeeze(angles)

        # Compute pos
        pos = np.zeros((n, 3))
        pos[:, 0] = distances * np.sin(angles)
        pos[:, 1] = distances * np.cos(angles)

        # Append UEs
        self.ue = UE(n, pos, gain, max_pow)

    def get_channels(self, tx_power, noise_power, codebook, direction=None, mask=None):
        """Calculate channel gains for the configuration estimation phase.

        Returns
        -------
        channel_gains_ce: ndarray of shape (num_ce_configs, num_ues)
            Downlink channel gains between the BS and UEs given each RIS configuration.
        """

        if mask is not None:
            ue_distances = self.ue.distances[mask]
            ue_angles = self.ue.angles[mask]
        else:
            ue_distances = self.ue.distances
            ue_angles = self.ue.angles

        if isinstance(ue_distances, float):
            ue_distances = np.array([ue_distances, ])
            ue_angles = np.array([ue_angles, ])

        if isinstance(codebook, float):
            codebook = np.array([codebook,])

        # Compute constant term
        num = self.bs.gain * self.ue.gain * (self.ris.size_el * self.ris.size_el)**2
        den = (4 * np.pi * self.bs.distance * ue_distances)**2

        # Compute pathloss component of shape (num_ues, )
        if direction == 'dl':
            pathloss = num / den * np.cos(self.bs.angle)**2
        elif direction == 'ul':
            pathloss = num / den * np.cos(ue_angles)**2

        # Compute propagation phase-shift
        propagation_angle = - (self.bs.distance + ue_distances -
        ((np.sin(self.bs.angle) - np.sin(ue_angles)) * ((self.ris.num_els_hor + 1) / 2) * self.ris.size_el))

        propagation_phase_shift = np.exp(1j * self.wavenumber * propagation_angle)

        # Define enumeration of the number of horizontal elements
        enumeration_num_els_hor = np.arange(1, self.ris.num_els_hor + 1)

        # Compute phase-shift contribution
        contribution = enumeration_num_els_hor[:, None, None] * (np.sin(ue_angles)[:, None] - np.sin(codebook[None, :]))

        # Compute array factors
        array_factor = np.exp(1j * self.wavenumber * self.ris.size_el * contribution)
        array_factor = array_factor.sum(axis=0)

        # Compute channels
        channels = np.sqrt(pathloss[:, None]) * propagation_phase_shift[:, None] * array_factor

        if direction == 'ul':
            channels = channels.conj()

        # Normalize channels
        channels *= np.sqrt(tx_power / noise_power)

        return channels












    # def get_ul_channels(self, codebook, ):
    #     """Calculate uplink channel gains.
    #
    #     Returns
    #     -------
    #     channel_gains_ul : ndarray of shape (num_access_configs, num_ues)
    #         Uplink channel gains between the BS and UEs given each RIS configuration.
    #     """
    #
    #     if mask is not None:
    #         ue_distances = self.ue.distances[mask]
    #         ue_angles = self.ue.angles[mask]
    #     else:
    #         ue_distances = self.ue.distances
    #         ue_angles = self.ue.angles
    #
    #     if isinstance(ue_distances, float):
    #         ue_distances = np.array([ue_distances, ])
    #         ue_angles = np.array([ue_angles, ])
    #
    #     if isinstance(codebook, float):
    #         codebook = np.array([codebook,])
    #
    #     # Compute constant term
    #     num = self.bs.gain * self.ue.gain * (self.ris.size_el * self.ris.size_el)**2
    #     den = (4 * np.pi * self.bs.distance * ue_distances)**2
    #
    #     # Compute DL pathloss component of shape (num_ues, )
    #
    #
    #     # Compute propagation phase-shift
    #     propagation_angle = - (self.bs.distance + ue_distances -
    #     ((np.sin(self.bs.angle) - np.sin(ue_angles)) * ((self.ris.num_els_hor + 1) / 2) * self.ris.size_el))
    #
    #     propagation_phase_shift = np.exp(-1j * self.wavenumber * propagation_angle)
    #
    #     # Define enumeration of the number of horizontal elements
    #     enumeration_num_els_hor = np.arange(1, self.ris.num_els_hor + 1)
    #
    #     # Compute phase-shift contribution
    #     contribution = enumeration_num_els_hor[:, None, None] * (np.sin(ue_angles)[:, None] - np.sin(codebook[None, :]))
    #
    #     # Compute array factors
    #     array_factor = np.exp(-1j * self.wavenumber * self.ris.size_el * contribution)
    #     array_factor = array_factor.sum(axis=0)
    #
    #     # Compute channel gains
    #     channel_gains = np.sqrt(pathloss[:, None]) * propagation_phase_shift[:, None] * array_factor
    #
    #     return channel_gains

    # def plot_scenario(self):
    #     """This method will plot the scenario of communication
    #     """
    #     # LaTeX type definitions
    #     rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
    #     rc('text', usetex=True)
    #     # Open axes
    #     fig, ax = plt.subplots()
    #
    #     # Box positioning
    #     box = plt.Rectangle((self.min_square_length, self.min_square_length), self.square_length, self.square_length, ec="black", ls="--", lw=1, fc='#45EF0605')
    #     ax.add_patch(box)
    #     # User positioning
    #     delta = self.min_square_length / 100
    #     # BS
    #     plt.scatter(self.bs.pos[0], self.bs.pos[1], c=common.node_color['BS'], marker=common.node_mark['BS'],
    #                 label='BS')
    #     # plt.text(self.bs.pos[:, 0], self.bs.pos[:, 1] + delta, s='BS', fontsize=10)
    #     # UE
    #     plt.scatter(self.ue.pos[:, 0], self.ue.pos[:, 1], c=common.node_color['UE'], marker=common.node_mark['UE'],
    #                 label='UE')
    #     for k in np.arange(self.ue.n):
    #         plt.text(self.ue.pos[k, 0], self.ue.pos[k, 1] + delta, s=f'{k}', fontsize=10)
    #     # RIS
    #     plt.scatter(self.ris.pos[0], self.ris.pos[1], c=common.node_color['RIS'], marker=common.node_mark['RIS'],
    #                 label='RIS')
    #     # plt.text(self.ris.pos[:, 0], self.ris.pos[:, 1] + delta, s='RIS', fontsize=10)
    #     # Set axis
    #     # ax.axis('equal')
    #     ax.set_xlabel('$x$ [m]')
    #     ax.set_ylabel('$y$ [m]')
    #     # limits
    #     ax.set_ylim(ymin=-self.min_square_length / 2)
    #     # Legend
    #     handles, labels = plt.gca().get_legend_handles_labels()
    #     by_label = OrderedDict(zip(labels, handles))
    #     ax.legend(by_label.values(), by_label.keys())
    #     # Finally
    #     plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
    #     plt.show(block=False)
