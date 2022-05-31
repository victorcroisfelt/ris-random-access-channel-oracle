import numpy as np
from scipy.constants import speed_of_light

import matplotlib.pyplot as plt
from matplotlib import rc

import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
#import cvxpy as cvx

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

def channel_model(bs_gain, bs_pos,
                  ue_gain, ue_pos,
                  ris_size_el, ris_num_els_hor, ris_num_els_ver, ris_configs):
    """Get Downlink (DL) and Uplink (UL) channel gain.

    Returns
    -------
    channel_gains_dl : ndarray of shape (num_configs, num_ues)
        Downlink channel gain between the BS and each UE for each RIS configuration.

    channel_gains_ul : ndarray of shape (num_configs, num_ues)
        Uplink channel gain between the BS and each UE for each RIS configuration.

    """

    # Extract distances and angles
    bs_distance = np.linalg.norm(bs_pos)
    bs_angle = np.arctan2(bs_pos[1], bs_pos[0])

    ue_distances = np.linalg.norm(ue_pos, axis=0)
    ue_angles = np.arctan2(ue_pos[1, ], ue_pos[0, :])

    # Compute constant term
    num = bs_gain * ue_gain * (ris_size_el * ris_size_el)**2
    den = (4 * np.pi * bs_distance * ue_distances)**2

    const = num/den
    breakpoint()
    # Compute DL pathloss component of shape (num_ues, )
    pathloss_dl = const * np.cos(bs_angle)**2

    # Compute UL pathloss component of shape (num_ues, )
    pathloss_ul = const * np.cos(ue_angles)**2

    # Compute constant phase component of shape (num_ues, )
    distances_sum = (bs_distance + ue_distances)
    disagreement = (np.sin(bs_angle) - np.sin(ue_angles)) * ((ris_num_els_hor + 1) / 2) * ris_size_el

    phi = - wavenumber * (distances_sum - disagreement)
    breakpoint()
    # Compute array factor of shape (num_configs, num_ues)
    enumeration_num_els_x = np.arange(1, ris_num_els_hor + 1)
    sine_differences = (np.sin(ue_angles[np.newaxis, :, np.newaxis]) - np.sin(ris_configs[:, np.newaxis, np.newaxis]))

    argument = wavenumber * sine_differences * enumeration_num_els_x[np.newaxis, np.newaxis, :] * ris_size_el

    array_factor_dl = ris_num_els_ver * np.sum(np.exp(+1j * argument), axis=-1)
    array_factor_ul = array_factor_dl.conj()

    # Compute channel gains of shape (num_configs, num_ues)
    channel_gains_dl = np.sqrt(pathloss_dl[np.newaxis, :]) * np.exp(+1j * phi[np.newaxis, :]) * array_factor_dl
    channel_gains_ul = np.sqrt(pathloss_ul[np.newaxis, :]) * np.exp(-1j * phi[np.newaxis, :]) * array_factor_ul

    return channel_gains_dl, channel_gains_ul

########################################
# Parameters
########################################

#-----
# Eletromagnetics
#-----

# Signal
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency
wavenumber = 2 * np.pi / wavelength

# Distances
#maximum_distance = 100

# Noise
noise_power = 10 ** (-94.0 / 10)  # mW

#-----
# RIS
#-----

# Number of configurations
ris_num_configs = 4

# Number of elements
ris_num_els_ver = ris_num_els_hor = 10  # vertical/horizontal

# Size of each element
ris_size_el = wavelength

# RIS size along one of the dimensions
ris_size = ris_num_els_ver * ris_size_el

# RIS configurations
ris_angular_resolution = ((np.pi / 2) - 0) / ris_num_configs
ris_configs = np.arange(ris_angular_resolution / 2, np.pi / 2, ris_angular_resolution)

print(ris_configs)

# Minimum distant to RIS
minimum_distance = (2/wavelength) * ris_size**2

#-----
# BS
#-----

# BS antenna gain
bs_gain = 10**(5/10)

# Position
bs_angle = np.deg2rad([-45.0])
bs_pos = minimum_distance * np.array([np.sin(bs_angle), np.cos(bs_angle)])

#-----
# UE
#-----

# UE antenna gain
ue_gain = 10**(5/10)

# Define specific positions for the UEs
ue_angles_deg = np.array([0.0, 30.0, 60.0, 90.0])
#ue_angles_deg = np.array([45.0])
ue_angles = np.deg2rad(ue_angles_deg)

# Prepare to save UE positions
ue_pos = np.zeros((3, ue_angles_deg.size))

# Compute UE positions
ue_pos[0, :] = np.sin(ue_angles)
ue_pos[1, :] = np.cos(ue_angles)

ue_pos *= minimum_distance

# Get channel gains for the UEs
channel_gains_dl, channel_gains_ul = channel_model(bs_gain, bs_pos, ue_gain, ue_pos, ris_size_el, ris_num_els_hor, ris_num_els_ver, ris_configs)

