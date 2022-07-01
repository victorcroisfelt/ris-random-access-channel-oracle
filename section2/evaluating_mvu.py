# -*- coding: utf-8 -*-
"""shannon-nyquist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rrWQ-d-wkfkvCnTX6grWVn6lpai5unHP
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

from scipy.constants import speed_of_light
from scipy.fft import fft, fftfreq, fftshift
from scipy import interpolate

from tqdm import trange

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb}')
matplotlib.rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 8})

########################################
# Private functions
########################################


def channel_model(bs_gain, bs_pos,
                  ue_gain, ue_pos,
                  ris_size_el, ris_num_els_hor, ris_num_els_ver, ris_configs
                  ):
    """Get downlink channel gains.

    Returns
    -------
    channel_gains_dl : ndarray of shape (num_configs, num_ues)

    """

    # Extract distances and angles
    bs_distance = np.linalg.norm(bs_pos)
    bs_angle = np.arctan2(bs_pos[0], bs_pos[1])

    ue_distances = np.linalg.norm(ue_pos, axis=0)
    ue_angles = np.arctan2(ue_pos[0, :], ue_pos[1, :])

    # Compute DL pathloss of shape (num_ues, )
    num = bs_gain * ue_gain * (ris_size_el * ris_size_el)**2
    den = (4 * np.pi * bs_distance * ue_distances)**2

    const = num/den

    pathloss_dl = const * np.cos(bs_angle)**2

    # Compute fundamental frequency
    fundamental_freq = ris_size_el / wavelength

    # Compute term 1
    term1 = np.sqrt(pathloss_dl) * ris_num_els_ver

    # Compute term 2
    term2 = np.exp(1j * 2 * np.pi * fundamental_freq * ((bs_distance + ue_distances) / ris_size_el))

    # Compute term 3
    term3 = np.exp(-1j * 2 * np.pi * fundamental_freq * (ris_num_els_hor + 1) / 2 * (np.sin(bs_angle) - np.sin(ue_angles)))

    # Compute term 4
    enumeration_num_els_x = np.arange(1, ris_num_els_hor + 1)

    term4 = np.exp(1j * 2 * np.pi * fundamental_freq * enumeration_num_els_x[:, None, None] * (np.sin(ue_angles)[:, None] - np.sin(ris_configs)[None, :]))
    term4 = term4.transpose(1, 0, 2)

    term4 = term4[:, :, :].sum(axis=1)

    # Compute channel gains
    channel_gains = term1[:, None] * term2[:, None] * term3[:, None] * term4

    return channel_gains

########################################
# System Parameters
########################################


# Electromagnetics
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency
wavenumber = 2 * np.pi / wavelength

# Define maximum distance
maximum_distance = 100

# RIS

# Number of elements
ris_num_els_ver = ris_num_els_hor = 10  # vertical/horizontal

# Size of each element
ris_size_el = wavelength/2

# RIS size along one of the dimensions
ris_size = ris_num_els_ver * ris_size_el

# Minimum distance to the RIS (far-field)
minimum_distance = (2 / wavelength) * ris_size**2

# BS

# BS antenna gain
bs_gain = 10**(5/10)

# BS position
bs_angle = np.deg2rad([-45.0])
bs_pos = minimum_distance * np.array([np.sin(bs_angle), np.cos(bs_angle)])

########################################
# Generating UEs
########################################

# Define number of UEs
num_ues = 100

# UE antenna gain
ue_gain = 10**(5/10)

# Generate distances
ue_distances = np.sqrt(np.random.rand(num_ues) * (maximum_distance ** 2 - minimum_distance ** 2) + minimum_distance ** 2)

# Generate angles
ue_angles = (np.pi / 2) * np.random.rand(num_ues)

# Get UE positions
ue_pos = np.zeros((3, num_ues))

# Compute UE positions
ue_pos[0, :] = np.sin(ue_angles)
ue_pos[1, :] = np.cos(ue_angles)

ue_pos *= ue_distances

########################################
# Generating channel gains
########################################

# Number of configurations or samples
ris_num_configs = 100

# Signal range
signal_range = (np.pi/2 - 0)

# Sampling period and frequency
sampling_period = signal_range/ris_num_configs
sampling_frequency = 1/sampling_period

# Spatial step
ris_configs = np.linspace(0, signal_range, ris_num_configs)

# Generate signal
channel_gains = channel_model(
    bs_gain, bs_pos,
    ue_gain, ue_pos,
    ris_size_el, ris_num_els_hor, ris_num_els_ver, ris_configs)

########################################
# Simulation parameters
########################################

# Define range of MVU error tolerance
tolerances = np.array([0.1, 0.01, 0.001])

# Define SNR range
snr_db_range = np.linspace(-10, 10, 11)

# Transform SNR linear
snr_range = 10**(snr_db_range/10)

# Number of noise realizations
num_noise_realizations = 100

# Prepare to save MVU MSEs
mse_mvu = np.zeros((tolerances.size, snr_range.size, num_ues))

# Go through all SNR values
for ss in trange(snr_range.size, desc='simulating', unit="snr points"):

    # Extract current SNR
    snr = snr_range[ss]

    # Go through all tolerances
    for tt, tolerance in enumerate(tolerances):

        # Compute the number of channel uses
        num_channel_uses_ce = int(np.ceil(1 / (snr * tolerance)))

        # Define reference signal
        reference_signal = (1 / np.sqrt(2)) * (np.ones((num_noise_realizations, num_channel_uses_ce)) + 1j * np.ones((num_noise_realizations, num_channel_uses_ce)))

        # Go through all UEs
        for ue in range(num_ues):

            # Prepare to save estimated channel gains
            estimated_channel_gains = np.zeros((ris_num_configs, num_noise_realizations), dtype=np.complex_)

            # Go through configurations
            for ce in range(ris_num_configs):

                # Generate noise
                noise = (1 / np.sqrt(2)) * (np.random.randn(num_noise_realizations, num_channel_uses_ce) + 1j * np.random.randn(num_noise_realizations, num_channel_uses_ce))

                # Generate received signals
                received_signals = np.sqrt(snr) * channel_gains[ue, ce, None, None] * reference_signal[:, :] + noise[:, :]

                # Prepare to save MVU estimates
                estimated_channel_gains[ce, :] = (1/(num_channel_uses_ce * np.sqrt(snr))) * (reference_signal.conj() * received_signals).sum(axis=-1)

            # Compute MSE
            mse_mvu[tt, ss, ue] = np.mean((np.abs(estimated_channel_gains - channel_gains[ue, :, None])**2))

########################################
# Plot reconstruction MSE
########################################
fig, ax = plt.subplots(figsize=(3.15, 3))

styles = ['--', '-.', ':']
labels = ['10^{-1}', '10^{-2}', '10^{-3}']

# Go through all tolerances
for tt, tolerance in enumerate(tolerances):
    ax.plot(snr_db_range, np.mean(mse_mvu[tt, :], axis=-1), linestyle=styles[tt], color='black',  label=r'$\mathrm{tol}=' + str(labels[tt]) + '$')

    ax.fill_between(snr_db_range, np.percentile(mse_mvu[tt, :], 25, axis=-1), np.percentile(mse_mvu[tt, :], 75, axis=-1))

ax.set_xlabel(r'$\mathrm{SNR}^{\rm DL}$ [dB]')
ax.set_ylabel(r'MSE MVU')

ax.legend(fontsize='x-small', loc='best')

ax.set_yscale('log')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()