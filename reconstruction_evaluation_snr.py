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

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb}')

# matplotlib.rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 12})
#
# matplotlib.rc('xtick', labelsize=8)
# matplotlib.rc('ytick', labelsize=8)
#
# matplotlib.rc('text', usetex=True)
#
# matplotlib.rcParams['text.latex.preamble'] = [
#     r'\usepackage{amsfonts}',
#     r'',
#     r''
# ]


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

    # Compute DL pahloss of shape (num_ues, )
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
    term3 = np.exp(-1j  * 2 * np.pi * fundamental_freq * (ris_num_els_hor + 1) / 2 * (np.sin(bs_angle) - np.sin(ue_angles)))

    # Compute term 4
    enumeration_num_els_x = np.arange(1, ris_num_els_hor + 1)

    term4 = np.exp(1j * 2 * np.pi * fundamental_freq * enumeration_num_els_x[:, None, None] * (np.sin(ue_angles)[:, None] - np.sin(ris_configs)[None, :]))
    term4 = term4.transpose(1, 0, 2)

    return term1, term2, term3, term4

########################################
# Parameters
########################################

# Electromagnetics
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency
wavenumber = 2 * np.pi / wavelength

# RIS

# Number of elements
ris_num_els_ver = ris_num_els_hor = 10  # vertical/horizontal

# Size of each element
ris_size_el = wavelength/2

# RIS size along one of the dimensions
ris_size = ris_num_els_ver * ris_size_el

# Minimum distance to the RIS
minimum_distance = (2 / wavelength) * ris_size**2

# BS

# BS antenna gain
bs_gain = 10**(5/10)

# BS position
bs_angle = np.deg2rad([-45.0])
bs_pos = minimum_distance * np.array([np.sin(bs_angle), np.cos(bs_angle)])

# UE

# UE antenna gain
ue_gain = 10**(5/10)

# Define specific positions for the UEs
#ue_angles_deg = np.array([30.0, 45.0, 60.0])
ue_angles_deg = np.array([45.0])
ue_angles = np.deg2rad(ue_angles_deg)

# Prepare to save UE positions
ue_pos = np.zeros((3, ue_angles_deg.size))

# Compute UE positions
ue_pos[0, :] = np.sin(ue_angles)
ue_pos[1, :] = np.cos(ue_angles)

ue_pos *= minimum_distance

########################################
# Generating signal
########################################

# Compute fundamental frequency
fundamental_frequency = ris_size_el / wavelength

# Number of configurations or samples
ris_num_configs = 1001

# Spatial duration of the signal
spatial_duration = (np.pi/2 - 0)

# Sampling period and frequency
sampling_period = spatial_duration/ris_num_configs
sampling_frequency = 1/sampling_period

# Spatial step
spatial_step = np.linspace(0, spatial_duration, ris_num_configs)

# Generate signal
term1, term2, term3, term4 = channel_model(
    bs_gain, bs_pos,
    ue_gain, ue_pos,
    ris_size_el, ris_num_els_hor, ris_num_els_ver, spatial_step)

signal = term4[0, :, :].sum(axis=0)

########################################
# Sampling signal
########################################

# Number of configurations or samples
num_samples = int(np.ceil(2 * np.pi * ris_num_els_ver * fundamental_frequency))

# Spatial duration of the signal
spatial_duration = (np.pi/2 - 0)

# Sampling period and frequency
sampling_period = spatial_duration/num_samples
sampling_frequency = 1/sampling_period

# Spatial step
spatial_step_sampled = np.linspace(0, spatial_duration, num_samples)

# Generate signal
term1, term2, term3, term4 = channel_model(
    bs_gain, bs_pos,
    ue_gain, ue_pos,
    ris_size_el, ris_num_els_hor, ris_num_els_ver, spatial_step_sampled)

signal_sampled = term4[0, :, :].sum(axis=0)

########################################
# PLot signal
########################################
fig, axes = plt.subplots(2, 1)

axes[0].plot(np.rad2deg(spatial_step), signal.real, color='black')
axes[1].plot(np.rad2deg(spatial_step), signal.imag, color='black')

axes[0].stem(np.rad2deg(spatial_step_sampled), signal_sampled.real)
axes[1].stem(np.rad2deg(spatial_step_sampled), signal_sampled.imag)

axes[0].set_xlabel(r'config. angle $\theta_s$ in degrees')
axes[1].set_xlabel(r'config. angle $\theta_s$ in degrees')

axes[0].set_ylabel(r'$\mathfrak{Re}(a_k(\theta_s))$')
axes[1].set_ylabel(r'$\mathfrak{Im}(a_k(\theta_s))$')

axes[0].set_xticks(np.arange(0, 90, 10))
axes[1].set_xticks(np.arange(0, 90, 10))

# axes[0].legend(fontsize='x-small', framealpha=0.5)

axes[0].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

########################################
# Frequency analysis
########################################

# Compute FFT
signal_freq = fft(signal_sampled)

# Compute frequency x-axis
freq = fftfreq(num_samples, d=sampling_period)

# Prepare for plot
freq_plot = fftshift(freq)
signal_freq_plot = fftshift(signal_freq)

# # Plot FFT
# fig, ax = plt.subplots()
#
# ax.plot(freq_plot, (1 / num_samples) * np.abs(signal_freq_plot))

########################################
# Signal reconstruction w/o estimates
########################################
# Apply zero-order reconstruction
zero_order_f_real = interpolate.interp1d(spatial_step_sampled, signal_sampled.real)
zero_order_f_imag = interpolate.interp1d(spatial_step_sampled, signal_sampled.imag)

# Get values
zero_order_real = zero_order_f_real(spatial_step)
zero_order_imag = zero_order_f_imag(spatial_step)

# Reconstructed signal
reconstructed_signal = zero_order_real + 1j * zero_order_imag

# Evaluate MSE
mse_reconstruction_true = (np.abs(signal - reconstructed_signal) ** 2).mean()

########################################
# Signal reconstruction
########################################
fig, axes = plt.subplots(2, 1)

axes[0].plot(np.rad2deg(spatial_step), signal.real, color='black')
axes[1].plot(np.rad2deg(spatial_step), signal.imag, color='black')

axes[0].plot(np.rad2deg(spatial_step_sampled), zero_order_f_real(spatial_step_sampled), linestyle=':')
axes[1].plot(np.rad2deg(spatial_step_sampled), zero_order_f_imag(spatial_step_sampled), linestyle=':')

axes[0].set_xlabel(r'config. angle $\theta_s$ in degrees')
axes[1].set_xlabel(r'config. angle $\theta_s$ in degrees')

axes[0].set_ylabel(r'$\mathfrak{Re}(a_k(\theta_s))$')
axes[1].set_ylabel(r'$\mathfrak{Im}(a_k(\theta_s))$')

axes[0].set_xticks(np.arange(0, 90, 10))
axes[1].set_xticks(np.arange(0, 90, 10))

# axes[0].legend(fontsize='x-small', framealpha=0.5)

axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

########################################
# Signal reconstruction
########################################

# Define tolerance
tolerances = np.array([0.1, 0.01, 0.001])

# Define SNR range
snr_db_range = np.linspace(-10, 10, 101)

# Transform SNR linear
snr_range = 10**(snr_db_range/10)

# Number of realizations
num_realizations = 100

# Save MSEs
mse_estimate = np.zeros((tolerances.size, snr_range.size, num_realizations))
mse_reconstruction = np.zeros((tolerances.size, snr_range.size, num_realizations))

# Go through all tolerances
for tt, tolerance in enumerate(tolerances):

    # Go through all SNR values
    for ss, snr in enumerate(snr_range):

        # Compute the number of channel uses
        num_channel_uses_ce = int(np.ceil(1 / (snr * tolerance)))

        # Define reference signal
        reference_signal = (1/np.sqrt(2)) * (np.ones((num_channel_uses_ce, num_realizations)) + 1j * np.ones((num_channel_uses_ce, num_realizations)))

        # Generate noise
        noise = (1/np.sqrt(2)) * (np.random.randn(num_channel_uses_ce, num_realizations) + 1j * np.random.randn(num_channel_uses_ce, num_realizations))

        # Generate received signals
        received_signals = np.sqrt(snr) * signal_sampled[:, None, None] * reference_signal[None, :, :] + noise[None, :, :]

        # Prepare to save MVU estimates
        estimated_channel_gains = np.zeros((num_samples, num_realizations), dtype=np.complex_)

        # Go through all the CE slots
        for ce in range(num_samples):

            estimated_channel_gains[ce, :] = 1/(num_channel_uses_ce * np.sqrt(snr)) * (reference_signal.conj() * received_signals[ce, :, :]).sum(axis=0)

        # Compute MSE estimate
        mse_estimate[tt, ss, :] = (np.abs(estimated_channel_gains - signal_sampled[:, None])**2).mean(axis=0)

        # Go through each realization
        for rr in range(num_realizations):

            # Apply zero-order reconstruction
            zero_order_f_real = interpolate.interp1d(spatial_step_sampled, estimated_channel_gains[:, rr].real)
            zero_order_f_imag = interpolate.interp1d(spatial_step_sampled, estimated_channel_gains[:, rr].imag)

            # Get values
            zero_order_real = zero_order_f_real(spatial_step)
            zero_order_imag = zero_order_f_imag(spatial_step)

            # Reconstructed signal
            reconstructed_signal = zero_order_real + 1j * zero_order_imag

            # Evaluate MSE
            mse_reconstruction[tt, ss, rr] = (np.abs(signal - reconstructed_signal)**2).mean()

########################################
# Plot
########################################
fig, ax = plt.subplots(figsize=(3.15, 3))

styles = ['--', '-.', ':']
labels = ['10^{-1}', '10^{-2}', '10^{-3}']

# Go through all tolerances
for tt, tolerance in enumerate(tolerances):
    ax.plot(snr_db_range, np.mean(mse_estimate[tt, :, :], axis=-1), linestyle=styles[tt], color='black',  label=r'$\mathrm{tol}=' + str(labels[tt]) + '$')

ax.set_xlabel(r'$\mathrm{SNR}^{\rm DL}$ [dB]')
ax.set_ylabel(r'estimate MSE')

ax.legend(fontsize='x-small', loc='best')

ax.set_yscale('log')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

fig, ax = plt.subplots(figsize=(3.15, 3))

ax.plot(snr_db_range, mse_reconstruction_true * np.ones_like(snr_db_range), linewidth=1.5, linestyle='-', color='black', label='true')

# Go through all tolerances
for tt, tolerance in enumerate(tolerances):
    ax.plot(snr_db_range, np.mean(mse_reconstruction[tt, :, :], axis=-1), linewidth=1.5, linestyle=styles[tt], color='black',  label=r'$\mathrm{tol}=' + str(labels[tt]) + '$')

ax.set_xlabel(r'$\mathrm{SNR}^{\rm DL}$ [dB]')
ax.set_ylabel(r'reconstruction MSE')

ax.legend(fontsize='x-small', framealpha=0.5)

ax.set_yscale('log')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()


# plt.show()
#
#
# #-----
# # BS
# #-----
#
#
#
# #-----
# # UE
# #-----
#
#
#
#
#
# # Get constant term
# constant = term1 * term2 * term3
#
#
#
#
# # Define Fourier terms
# fourier_terms = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
#
# complex_coeffs = fourier_complex_coeff(ue_pos, ris_size_el, ris_num_els_hor, ris_configs, fourier_terms)
#
# fig, ax = plt.subplots()
#
# ax.stem(fourier_terms, np.abs(np.squeeze(complex_coeffs))**2)
#
# plt.show()

