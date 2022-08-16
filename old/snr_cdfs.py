from environment.box import Box

import numpy as np
from scipy.constants import speed_of_light

import matplotlib.pyplot as plt
from matplotlib import rc

import time

########################################
# General parameters
########################################
seed = 42
np.random.seed(seed)

# Define maximum distance
maximum_distance = 100

# Noise power
noise_power = 10 ** (-94.0 / 10)  # mW

########################################
# Simulation parameters
########################################

# Range of configurations
num_configs_range = np.array([2, 4, 8])

# Probability of access
prob_access = 0.001

# Number of setups
num_setups = 10000

# Range of number of elements
num_els_range = np.array([2, 4, 8, 12, 16])

# Size of each element
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

size_el = wavelength

# Compute minimum distance
ris_size = num_els_range.max() * size_el
minimum_distance = (2 / wavelength) * ris_size ** 2

########################################
# Simulation
########################################

# Prepare to save results
snrs_ul_general = {num_els: {} for num_els in num_els_range}

#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS (fixed)
box.place_bs(distance=25, zenith_angle_deg=45)

# Place UEs
box.place_ue(int(num_setups))

# Go through all different number of elements
for nn, num_els in enumerate(num_els_range):

    # Prepare to store Uplink SNRs
    snrs_ul = {config: {} for config in num_configs_range}

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        # Place RIS
        box.place_ris(num_configs=num_configs, num_els_z=num_els, num_els_x=num_els)

        # Get channel gains
        channel_gains_dl, channel_gains_ul = box.get_channel_model()

        # Generate noise vector at BS
        noise_bs = np.sqrt(noise_power / 2) * (np.random.randn(num_configs) + 1j * np.random.randn(num_configs))

        # Prepare to save UL SNRs of shape (num_configs, num_pilots)
        snrs = np.zeros((num_configs, num_setups))

        # Go through all access frames
        for aa in range(num_configs):

            # Go through all UEs
            for ue in range(num_setups):

                # Compute UL received signal response
                snrs[aa, ue] = box.ue.max_pow * np.abs(channel_gains_ul[aa, ue])**2 / np.abs(noise_bs[aa])**2

        # Sort SNRs
        snrs_sorted = -np.sort(-snrs, axis=0)

        # Go through all access frames
        for aa in range(num_configs):
            snrs_ul[num_configs][aa] = list(snrs_sorted[aa])

    # Store general results
    snrs_ul_general[num_els] = snrs_ul

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    # Open axes
    fig, axes = plt.subplots(ncols=5)

    # Go through all different number of elements
    for nn, num_els in enumerate(num_els_range):

        # Go through all access frames
        for aa in range(num_configs):

            # Extract data
            data = np.sort(snrs_ul_general[num_els][num_configs][aa])

            axes[nn].plot(10 * np.log10(data), np.linspace(0, 1, num=data.size), linewidth=2.0, label='$S=' + str(aa) + '$')

        # Set axis
        axes[nn].set_xlabel('UL SNR [dB]')
        axes[nn].set_ylabel('CDF')

        axes[nn].set_title('$N_x=N_z=' + str(num_els) + '$')

        # Legend
        axes[nn].legend()

        axes[nn].grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

    # Finally
    plt.tight_layout()
    plt.show(block=False)


# # Processed rx powers
# rx_powers_proc = {config: [] for config in num_configs_range}
#
# # Go through all number of configurations
# for cc, num_configs in enumerate(num_configs_range):
#     rx_powers_proc[num_configs] = np.array([item for sublist in rx_powers[num_configs] for item in sublist])

