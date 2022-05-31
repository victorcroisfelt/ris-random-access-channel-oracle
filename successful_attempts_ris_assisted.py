from environment.box import Box

import numpy as np
from scipy.constants import speed_of_light

from randomaccessfunc import collision_resolution

from tqdm import trange

########################################
# Preamble
########################################
seed = 42
np.random.seed(seed)

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

# Range of configurations
num_configs_range = np.arange(1, 21)

# Range of number of contending UEs
num_ues_range = np.arange(1, 11)

# Minimum threshold value
gamma_th = 1

# Number of setups
num_setups = int(1e4)

# Methods to choose configuration
choose_config = 'strongest'
#choose_config = 'rand'

########################################
# Simulation
########################################
print('--------------------------------------------------')
print('RIS-assisted')
print('Method: ' + choose_config)
print('--------------------------------------------------')

# Prepare to store number of successful attempts
total_num_successful_attempts = np.empty((num_configs_range.size, num_ues_range.size, num_setups))
total_num_successful_attempts[:] = np.nan


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=25, zenith_angle_deg=45)

# Go through all setups
for ss in trange(num_setups, desc="Simulating", unit="setups"):

    # Go through all different total number of UEs
    for ii, num_ues in enumerate(num_ues_range):

        # Store enumeration of active UEs
        enumeration_active_ues = np.arange(0, num_ues).astype(int)

        # Place UEs
        box.place_ue(int(num_ues))

        # Go through all number of configurations
        for cc, num_configs in enumerate(num_configs_range):

            # Store enumeration of configs/access frames
            enumeration_configs = np.arange(0, num_configs).astype(int)

            # Place RIS
            box.place_ris(num_configs=num_configs, num_els_z=Nz, num_els_x=Nx)

            # Obtain channel gains
            channel_gains_dl, channel_gains_ul = box.get_channel_model()

            ##################################################
            ## DL - Training phase
            ##################################################

            # Compute configurations' strength of shape (num_configs, num_ues)
            configs_strength = np.sqrt(box.ue.max_pow) * np.abs(channel_gains_dl)

            ##################################################
            ## Deciding access frames to transmit
            ##################################################

            # Define number of access frames
            num_access_frames = num_configs

            # Create a dictionary to store UE choices
            ue_choices = {}

            # Choose access frames
            if choose_config == 'strongest':

                # Prepare to save the strongest access frames for each UE
                strongest_access_frames = np.zeros((1, num_ues)).astype(int)

                # Select the strongest access frame for each UE
                strongest_access_frames[:] = np.argmax(configs_strength, axis=0)

                # Store choices
                ue_choices = {ue: list(strongest_access_frames[:, ue]) for ue in enumeration_active_ues}

            elif choose_config == 'rand':

                # Prepare to save access frames probabilities for each UE
                access_frame_probabilities = np.zeros((num_access_frames, num_ues))

                # Compute vector of probabilities
                access_frame_probabilities[:] = configs_strength / configs_strength.sum(axis=0)[np.newaxis, :]

                # Go through all active UEs
                for ue in enumeration_active_ues:

                    # Prepare to save selected access frames
                    selected = np.zeros(num_access_frames).astype(int)

                    # Keep flipping
                    while True:

                        # Flipping coins
                        tosses = np.random.rand(num_access_frames)

                        # Selected access frames
                        selected[tosses <= access_frame_probabilities[:, ue]] = True

                        # Store list of choices
                        choices = [access_frame for access_frame in enumeration_configs if selected[access_frame] == 1]

                        if len(choices) >= 1:
                            break

                    # Store choices
                    ue_choices[ue] = choices

                    del choices

            ##################################################
            ## UL - Naive UEs' responses
            ##################################################

            # Generate noise vector at the BS
            noise_bs = np.sqrt(noise_power / 2) * (np.random.randn(num_access_frames) + 1j * np.random.randn(num_access_frames))

            # Store buffered UL received signal responses as a dictionary
            buffered_access_attempts = {access_frame: {} for access_frame in enumeration_configs}

            # Go through all access frames
            for aa in enumeration_configs:

                # Create another dictionary
                buffered_access_attempts[aa] = {}

                # Go through all active UEs
                for ue in enumeration_active_ues:

                    if aa in ue_choices[ue]:
                        buffered_access_attempts[aa][ue] = np.sqrt(box.ue.max_pow) * channel_gains_ul[aa, ue] * np.sqrt(1 / 2) * (1 + 1j)

                # Add noise
                buffered_access_attempts[aa]['noise'] = noise_bs[aa]

            ##################################################
            ## Collision resolution strategy
            ##################################################
            total_num_successful_attempts[cc, ii, ss] = collision_resolution(ue_choices, buffered_access_attempts, gamma_th)

# Save data
np.savez('data/ris_' + choose_config + '_N' + str(Nx * Nz) + '_contending_full.npz',
         num_configs_range=num_configs_range,
         num_ues_range=num_ues_range,
         total_num_successful_attempts=total_num_successful_attempts
         )