from environment.box import Box

import numpy as np
from scipy.constants import speed_of_light

from randomaccessfunc import collision_resolution_slotted

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

# Channel load range
channel_load_range = np.linspace(0, 10, 21)

# Minimum threshold value
gamma_th = 1

# Number of setups
num_setups = int(1e4)

# Define the access policy
access_policy = 'strongest'
#access_policy = 'rand'
#access_policy = 'aloha'

########################################
# Simulation
########################################
print('--------------------------------------------------')
print('RIS-assisted')
print('Access policy: ' + access_policy)
print('--------------------------------------------------')

# Prepare to store number of successful attempts
avg_num_successful_attempts = np.empty((num_configs_range.size, channel_load_range.size, num_setups))
avg_num_successful_attempts[:] = np.nan


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=25, zenith_angle_deg=45)

# Generate UEs transmissions per slot
#transmissions_ues = np.random.poisson(lam=channel_load_range, size=(num_setups, channel_load_range.size))

# Go through all different channel loads
for ii in trange(channel_load_range.size, desc="Simulating", unit="channel load"):

    # Extract channel load
    channel_load = channel_load_range[ii]

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        # Place RIS
        box.place_ris(num_configs=num_configs, num_els_z=Nz, num_els_x=Nx)

        # Define number of access slots
        num_access_slots = num_configs

        # Store enumeration of access slots
        enumeration_access_slots = np.arange(0, num_configs).astype(int)

        # Go through all setups
        for ss in range(num_setups):

            # Prepare to store UE channel gains
            ue_channel_gains = {}

            # Prepare to store UE policies
            ue_policies = {}

            ##################################################
            # Generating new UEs
            ##################################################

            # Generate new UEs
            num_new_ues = np.random.poisson(lam=channel_load, size=(num_configs, ))

            # Compute total number of UEs
            total_num_new_ues = int(num_new_ues.sum())

            if total_num_new_ues == 0:
                continue

            # Place UEs
            box.place_ue(int(total_num_new_ues))

            # Obtain channel gains
            channel_gains_dl, channel_gains_ul = box.get_channel_model()

            # Enumerate new UEs
            enumeration_total_new_ues = np.arange(0, total_num_new_ues)

            ##################################################
            ## Training phase
            ##################################################

            # Compute configurations' strength of shape (num_configs, num_ues)
            configs_strength = np.abs(channel_gains_dl)

            # Choose access frames
            if access_policy == 'strongest':

                # Prepare to save the strongest access frames for each UE
                strongest_access_frames = np.zeros((1, total_num_new_ues)).astype(int)

                # Select the strongest access frame for each UE
                strongest_access_frames[:] = np.argmax(configs_strength, axis=0)

                # Store choices
                for new_ue in enumeration_total_new_ues:
                    ue_policies[new_ue] = list(strongest_access_frames[:, new_ue])

            elif access_policy == 'rand':

                # Prepare to save access frames probabilities for each UE
                access_frame_probabilities = np.zeros((num_access_slots, total_num_new_ues))

                # Compute vector of probabilities
                access_frame_probabilities[:] = configs_strength / configs_strength.sum(axis=0)[np.newaxis, :]

                # Store choices
                for new_ue in enumeration_total_new_ues:
                    ue_policies[new_ue] = list(access_frame_probabilities[:, new_ue])

            ##################################################
            # Access phasea
            ##################################################

            # Prepare to store UE access attempts
            ue_access = {}

            # Go through all access slots
            for aa in enumeration_access_slots:

                # Extract current number of new UEs
                current_num_new_ues = num_new_ues[aa]

                if current_num_new_ues == 0:
                    continue

                # UEs that became active in this slot
                enumeration_new_ues = enumeration_total_new_ues[num_new_ues[:aa].sum():num_new_ues[:(aa+1)].sum()]

                # Go through all active UEs
                for ue in enumeration_new_ues:

                    # Choose access frames
                    if access_policy == 'strongest':

                        if ue_policies[ue] >= aa:
                            ue_access[ue] = ue_policies[ue]

                    elif access_policy == 'rand':

                        # Prepare to save selected access frames
                        selected = np.zeros(num_access_slots - aa).astype(int)

                        # Flipping coins
                        tosses = np.random.rand(num_access_slots - aa)

                        # Selected access frames
                        selected[tosses <= ue_policies[ue][aa:]] = True

                        # Store list of choices
                        choices = [access_slot for access_slot in enumeration_access_slots[aa:] if selected[access_slot - aa] == 1]

                        if len(choices) >= 1:

                            # Store choices
                            ue_access[ue] = choices

                        del choices

                    elif access_policy == 'aloha':

                        # Prepare to save selected access frames
                        selected = np.zeros(num_access_slots - aa).astype(int)

                        # Flipping coins
                        tosses = np.random.rand(num_access_slots - aa)

                        # Selected access frames
                        selected[tosses <= 0.5] = True

                        # Store list of choices
                        choices = [access_slot for access_slot in enumeration_access_slots[aa:] if selected[access_slot - aa] == 1]

                        if len(choices) >= 1:

                            # Store choices
                            ue_access[ue] = choices

                        del choices

            ##################################################
            ## Buffered signal
            ##################################################

            if len(ue_access.keys()) == 0:
                continue

            # Generate noise vector at the BS
            noise_bs = np.sqrt(noise_power / 2) * (np.random.randn(num_access_slots) + 1j * np.random.randn(num_access_slots))

            # Store buffered UL received signal responses as a dictionary
            buffered_access_attempts = {access_slot: {} for access_slot in enumeration_access_slots}

            # Go through all access slots
            for aa in enumeration_access_slots:

                # Create another dictionary
                buffered_access_attempts[aa] = {}

                # Go through all active UEs
                for ue in ue_access.keys():

                    if aa in ue_access[ue]:
                        buffered_access_attempts[aa][ue] = np.sqrt(box.ue.max_pow) * channel_gains_ul[aa, ue] * np.sqrt(1 / 2) * (1 + 1j)

                # Add noise
                buffered_access_attempts[aa]['noise'] = noise_bs[aa]

            ##################################################
            ## Collision resolution strategy
            ##################################################

            # Get number of successful attempts
            num_successful_attempts = collision_resolution_slotted(ue_access, buffered_access_attempts, gamma_th)

            # Average with respect to how much was sent
            avg_num_successful_attempts[cc, ii, ss] = num_successful_attempts/len(ue_access.keys())

# Save data
np.savez('data/ris_' + access_policy + '_N' + str(Nx * Nz) + '_slot_based.npz',
         num_configs_range=num_configs_range,
         channel_load_range=channel_load_range,
         avg_num_successful_attempts=avg_num_successful_attempts
         )