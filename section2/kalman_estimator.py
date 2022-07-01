import numpy as np
from scipy.constants import speed_of_light

from environment.box import Box
from environment.nodes import UE

import matplotlib.pyplot as plt
from matplotlib import rc

import scipy.optimize as spopt
import scipy.fftpack as spfft
import scipy.ndimage as spimg
import cvxpy as cvx

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)


class KalmanFilter:

    def __init__(self, state_trans_model, obs_model, process_cov, obs_noise_cov):
        self.state_trans_model = state_trans_model
        self.obs_model = obs_model
        self.process_cov = process_cov
        self.obs_noise_cov = obs_noise_cov

        self.z_ = []
        self.x_ = [0.0]
        self.p_ = [1.0]

    def measure(self, z):
        self.z_.append(z)

    def predict(self):
        x_past = self.x_[-1]
        p_past = self.p_[-1]

        x_pres = self.state_trans_model * x_past
        p_pres = self.state_trans_model**2 * p_past + self.process_cov

        self.x_.append(x_pres)
        self.p_.append(p_pres)

    def update(self):

        z = self.z_[-1]

        x_pres = self.x_[-1]
        p_pres = self.p_[-1]

        # Residual
        res = z - self.obs_model * x_pres

        # Kalman gain
        kg = p_pres * self.obs_model / (self.obs_model**2 * p_pres + self.obs_noise_cov)

        # Update
        x_pres_post = x_pres + kg * res
        p_pres_post = (1 - kg * self.obs_model) * p_pres

        self.x_[-1] = x_pres_post
        self.p_[-1] = p_pres_post

########################################
# General parameters
########################################

# Number of elements
Nz = 10  # vertical
Nx = 10  # horizontal

# Size of each element
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

size_el = wavelength

# RIS size along one of the dimensions
ris_size = Nx * size_el

# Distances
maximum_distance = 100
minimum_distance = (2/wavelength) * ris_size**2

# Noise power
noise_power = 10 ** (-94.0 / 10)  # mW

########################################
# Simulation parameters
########################################

# Number of configurations
num_configs = 4

# Number of UEs positions
num_positions = 1001

########################################
# Simulation
########################################

# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed=42))

# Place BS
box.place_bs(distance=25, zenith_angle_deg=45)

# Place UEs along the arc
ue_angles = np.linspace(0, np.pi/2, num_positions)

# Prepare to save UE positions
ue_pos = np.zeros((num_positions, 3))

# Compute UE positions
ue_pos[:, 0] = np.sin(ue_angles)
ue_pos[:, 1] = np.cos(ue_angles)

# Place UEs
box.ue = UE(num_positions, ue_pos)

# Place RIS
box.place_ris(num_configs=num_configs, num_els_z=Nz, num_els_x=Nx)

# Obtain channel gains
channel_gains_dl, channel_gains_ul = box.get_channel_model()

########################################
# Plot 01 - Magnitude/Phase
########################################
fig, axes = plt.subplots(2, 1)

# Create a marker vector
markers = ['-', '--', '-.', ':']

# Go through all configurations
for config in range(num_configs):
    label = r'$\theta_' + str(config) + '=' + str(np.round(np.rad2deg(box.ris.configs[config]), 2)) + '^{\circ}$'

    axes[0].plot(np.rad2deg(ue_angles), 20*np.log10(np.abs(channel_gains_dl[config])), markers[config], label=label)
    axes[1].plot(np.rad2deg(ue_angles), np.rad2deg(np.angle(channel_gains_dl[config])), markers[config])

axes[0].set_xlabel(r'UE angle $\theta_k$ in degrees')
axes[1].set_xlabel(r'UE angle $\theta_k$ in degrees')

axes[0].set_ylabel('DL channel gain - mag.')
axes[1].set_ylabel('DL channel gain - phase')

axes[0].set_xticks(np.arange(0, 100, 10))
axes[1].set_xticks(np.arange(0, 100, 10))

axes[0].legend(fontsize='x-small', framealpha=0.5)

axes[0].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()
#plt.show()

########################################
# Plot 02 - Distribution
########################################
fig, ax = plt.subplots()

# Create a marker vector
markers = ['-', '--', '-.', ':']

# Go through all configurations
for config in range(num_configs):
    label = r'$\theta_' + str(config) + '=' + str(np.round(np.rad2deg(box.ris.configs[config]), 2)) + '^{\circ}$'
    ax.plot(np.sort(20*np.log10(np.abs(channel_gains_dl[config]))), np.linspace(0, 1, num_positions), markers[config], label=label)

ax.set_xlabel(r'DL channel gain - mag.')
ax.set_ylabel('CDF')

# axes[0].set_xticks(np.arange(0, 100, 10))
# axes[1].set_xticks(np.arange(0, 100, 10))

ax.legend(fontsize='x-small', framealpha=0.5)

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()
#plt.show()

########################################
# Looking in a different way
########################################

# Define specific positions for the UEs
ue_angles_deg = np.array([0.0, 30.0, 60.0, 90.0])
ue_angles = np.rad2deg(ue_angles_deg)

# Prepare to save UE positions
ue_pos = np.zeros((ue_angles_deg.size, 3))

# Compute UE positions
ue_pos[:, 0] = np.sin(ue_angles)
ue_pos[:, 1] = np.cos(ue_angles)

# Re-place UEs
box.ue = UE(ue_angles_deg.size, ue_pos)

# Place RIS
box.place_ris(num_configs=1001, num_els_z=Nz, num_els_x=Nx)

# Obtain channel gains
channel_gains_dl, channel_gains_ul = box.get_channel_model()

########################################
# Plot 03 - Looking in a different way
########################################
fig, axes = plt.subplots(2, 1)

# Create a marker vector
markers = ['-', '--', '-.', ':']

# Go through all UEs
for ue in range(ue_angles_deg.size):

    label = r'$\theta_' + str(ue) + '=' + str(np.round(ue_angles_deg[ue], 2)) + '^{\circ}$'

    axes[0].plot(np.rad2deg(box.ris.configs), 20*np.log10(np.abs(channel_gains_dl[:, ue])), markers[ue], label=label)
    axes[1].plot(np.rad2deg(box.ris.configs), np.rad2deg(np.angle(channel_gains_dl[:, ue])), markers[ue])

axes[0].set_xlabel(r'config. angle $\theta_s$ in degrees')
axes[1].set_xlabel(r'config. angle $\theta_s$ in degrees')

axes[0].set_ylabel('DL channel gain - mag.')
axes[1].set_ylabel('DL channel gain - phase')

axes[0].set_xticks(np.arange(0, 100, 10))
axes[1].set_xticks(np.arange(0, 100, 10))

axes[0].legend(fontsize='x-small', framealpha=0.5)

axes[0].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()
#plt.show()

#################
#################
#################

# Take a single UE
ue_angle_deg = 30
ue_angle = np.rad2deg(ue_angle_deg)

# Prepare to save UE positions
ue_pos = np.zeros((1, 3))

# Compute UE positions
ue_pos[:, 0] = np.sin(ue_angle)
ue_pos[:, 1] = np.cos(ue_angle)

# Re-place UEs
box.ue = UE(1, ue_pos)

# Number of configurations
num_configs = 4

# Place RIS
box.place_ris(num_configs=num_configs, num_els_z=Nz, num_els_x=Nx)

# Obtain channel gains
channel_gains_dl, channel_gains_ul = box.get_channel_model()

configs_4 = box.ris.configs
channel_gains_dl_4 = channel_gains_dl

# Number of configurations
num_configs = 1001

# Place RIS
box.place_ris(num_configs=num_configs, num_els_z=Nz, num_els_x=Nx)

# Obtain channel gains
channel_gains_dl, channel_gains_ul = box.get_channel_model()

configs_1001 = box.ris.configs
channel_gains_dl_1001 = channel_gains_dl

fig, axes = plt.subplots(1, 2)

axes[0].plot(np.rad2deg(configs_4), np.abs(channel_gains_dl_4), linewidth=0.0, marker='x', markersize=12, linestyle=None, label='$S=4$')
axes[0].plot(np.rad2deg(configs_1001), np.abs(channel_gains_dl_1001), linewidth=2, linestyle='-', label='$S=1001$', color='black')

axes[1].plot(np.rad2deg(configs_4), np.angle(channel_gains_dl_4), linewidth=0.0, marker='x', markersize=12, linestyle=None)
axes[1].plot(np.rad2deg(configs_1001), np.angle(channel_gains_dl_1001), linewidth=2, linestyle='-', color='black')

axes[0].set_xlabel(r'config. angle $\theta_s$ in degrees')
axes[1].set_xlabel(r'config. angle $\theta_s$ in degrees')

axes[0].set_ylabel('DL channel gain - mag.')
axes[1].set_ylabel('DL channel gain - phase')

axes[0].set_xticks(np.arange(0, 100, 10))
axes[1].set_xticks(np.arange(0, 100, 10))

axes[0].legend(fontsize='x-small', framealpha=0.5)

axes[0].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)


plt.tight_layout()
#plt.show()

########################################
# Estimation process
########################################

# Number of configurations
num_configs = 1001

# Place RIS
box.place_ris(num_configs=num_configs, num_els_z=Nz, num_els_x=Nx)

# Take a single UE
ue_angle_deg = 30
ue_angle = np.rad2deg(ue_angle_deg)

# Prepare to save UE positions
ue_pos = np.zeros((1, 3))

# Compute UE positions
ue_pos[:, 0] = np.sin(ue_angle)
ue_pos[:, 1] = np.cos(ue_angle)

# Re-place UEs
box.ue = UE(1, ue_pos)

# Obtain channel gains
channel_gains_dl, channel_gains_ul = box.get_channel_model()

# Define noise power
noise_power = (np.abs(channel_gains_dl).mean())**2

# Received signals
z_s = channel_gains_dl + np.sqrt(noise_power / 2) * (np.random.randn(num_configs, 1) + 1j * np.random.randn(num_configs, 1))
z_s_list = (np.squeeze(z_s).tolist())

# Create a Kalman Filter instance
kalman = KalmanFilter(state_trans_model=1, obs_model=1, process_cov=1, obs_noise_cov=1)

# Go through all measurements
for config in range(num_configs):

    # Update measurement
    kalman.measure(z_s_list[config])

    # Predict step
    kalman.predict()

    # Update step
    kalman.update()

fig, axes = plt.subplots(1, 2)

axes[0].plot(np.rad2deg(box.ris.configs), np.abs(channel_gains_dl), linewidth=2,  linestyle='-', label='true')
axes[0].plot(np.rad2deg(box.ris.configs), np.abs(z_s), linestyle='--', label='observed', alpha=.5)
axes[0].plot(np.rad2deg(box.ris.configs), np.abs(kalman.x_[1:]), linestyle=':', label='filtered', alpha=.5)

axes[1].plot(np.rad2deg(box.ris.configs), np.angle(channel_gains_dl), linewidth=2, linestyle='-')
axes[1].plot(np.rad2deg(box.ris.configs), np.angle(z_s), linestyle='--', alpha=.5)
axes[1].plot(np.rad2deg(box.ris.configs), np.angle(kalman.x_[1:]), linestyle=':', alpha=.5)

axes[0].set_xlabel(r'config. angle $\theta_s$ in degrees')
axes[1].set_xlabel(r'config. angle $\theta_s$ in degrees')

axes[0].set_ylabel('DL channel gain - mag.')
axes[1].set_ylabel('DL channel gain - phase')

axes[0].set_xticks(np.arange(0, 100, 10))
axes[1].set_xticks(np.arange(0, 100, 10))

axes[0].legend(fontsize='x-small', framealpha=0.5)

axes[0].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)


plt.tight_layout()
#plt.show()


#####


num_configs_range = np.arange(2, 1002)

nmse = np.zeros(num_configs_range.size)

# Go through all measurements
for cc, num_configs in enumerate(num_configs_range):

    # Create a Kalman Filter instance
    kalman = KalmanFilter(state_trans_model=1, obs_model=1, process_cov=1, obs_noise_cov=1)

    # Place RIS
    box.place_ris(num_configs=num_configs, num_els_z=Nz, num_els_x=Nx)

    # Obtain channel gains
    channel_gains_dl, channel_gains_ul = box.get_channel_model()

    # Define noise power
    noise_power = (np.abs(channel_gains_dl).mean()) ** 2

    # Received signals
    z_s = channel_gains_dl + np.sqrt(noise_power / 2) * (
                np.random.randn(num_configs, 1) + 1j * np.random.randn(num_configs, 1))

    z_s_list = (np.squeeze(z_s).tolist())

    # Go through all measurements
    for config in range(num_configs):

        # Update measurement
        kalman.measure(z_s_list[config])

        # Predict step
        kalman.predict()

        # Update step
        kalman.update()

    nmse[cc] = np.linalg.norm(np.squeeze(channel_gains_dl) - kalman.x_[1:])**2 / np.linalg.norm(channel_gains_dl)**2

fig, ax = plt.subplots()

ax.plot(np.flip(num_configs_range), np.flip(nmse), linestyle='-', linewidth=2.0)

ax.set_xlabel(r'number of configurations (samples), $S$')
ax.set_ylabel(r'NMSE')

plt.tight_layout()
#plt.show()

########################################
# Pipelining...
########################################
num_configs = 1001


abs_channel = np.abs(channel_gains_dl)

num_samples = int(.1 * num_configs)

sampling_mask = np.random.choice(num_configs, num_samples, replace=False)
sampling_mask.sort()

observed_abs_channel = abs_channel[sampling_mask]

fig, ax = plt.subplots()

#ax.plot(range)


breakpoint()
