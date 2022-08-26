import numpy as np

from scipy import interpolate
from scipy.constants import speed_of_light

#from randomaccessfunc import collision_resolution_slotted

from tqdm import trange

from src.box import Box
from src.frame import Frame

import matplotlib
import matplotlib.pyplot as plt

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb}')
matplotlib.rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 8})

########################################
# Preamble
########################################
seed = 42
np.random.seed(seed)

########################################
# Define system setup
########################################

# Wave parameters
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

# Number of RIS elements
num_els_ver = 10  # vertical
num_els_hor = 10  # horizontal

# Size of each element
size_el = wavelength/2

# RIS size along one of the dimensions
ris_size = num_els_hor * size_el

# Distances
maximum_distance = 100
minimum_distance = (2/wavelength) * ris_size**2

# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

########################################
# RA frame parameters
########################################

# Define SNR
snr_db = 10
snr = 10**(snr_db/10)

# Define desired MVU estimator tolerance
error_tolerance = 10e-3

# Define number of training channel uses
num_channel_uses = int(np.ceil(1 / (snr * error_tolerance)))

# Minimum SNR threshold value for decoding
decoding_snr = 1

# Initialize frame
frame = Frame()

# Initialize training block
frame.init_training(tr_num_slots, num_channel_uses, 0, decoding_snr)

########################################
# Simulation
########################################

# Number of setups
num_setups = int(1e4)

# Range of channel load
channel_loads = np.arange(1, 11)

# Range of reconstruction MSE error
range_rec_error = 10**(-np.arange(1, 4, dtype=np.double))

# Define the access policies
access_policies = ['RCURAP', 'RCARAP', 'RGSCAP', 'SMAP']

# Prepare to store number of successful attempts
avg_num_successful_attempts = np.zeros((len(access_policies), range_rec_error.size, channel_loads.size, num_setups))


#####


# Go through all access policies
for ap in trange(len(access_policies), desc="Access Policy", unit=" ap"):

    # Go through all channel loads
    for cc in trange(channel_loads.size, desc="Channel Load", unit=" chnload"):

        # Extract current channel load
        channel_load = channel_loads[cc]

        # Initialize access block
        frame.init_access(channel_load, num_channel_uses, 0, decoding_snr)

        # Generating new UEs
        range_num_ues = np.random.poisson(channel_load, (num_setups))

        # Go through all setups
        for ss in range(num_setups):

            # Extract current number of UEs
            num_ues = range_num_ues[ss]

            if num_ues == 0:

                avg_num_successful_attempts[ap, cc, ss] = np.nan

                continue

            # Enumerate UEs
            enumeration_ues = np.arange(0, num_ues)

            # Place UEs
            box.place_ue(num_ues)

            # True access info
            ac_true_info = box.get_ul_channel_gains(frame.ac.codebook)

            # Go through all recontruction errors
            for re in range(range_rec_error.size):

                # Extract current recontruction error
                rec_error = 

                # Generate noise
                noise = np.random.randn(num_setups, frame.ac.num_slots) + 1j * np.random.randn(num_setups, frame.ac.num_slots)
                noise *= np.sqrt(1/2) * np.sqrt(range_rec_error)

                # Noisy access info
                ac_info =

            ##################################################
            # 2) UL access block
            ##################################################

            # Apply access policy
            chosen_access_slots = frame.ac.access_policy(snr, ac_info, num_repetitions=num_repetitions, policy=access_policies[ap])

            # Prepare to save access attempts
            buffered_access_attempts = np.zeros((frame.ac.num_slots, frame.ac.num_channel_uses), dtype=np.complex_)

            # Generate messages
            messages = np.sqrt(1/2) * (np.random.randn(num_ues, frame.ac.num_channel_uses) + 1j * np.random.randn(num_ues, frame.ac.num_channel_uses))

            # Go through each access slot
            for ac_slot in chosen_access_slots.keys():

                # Extract number of colliding UEs
                colliding_ues = chosen_access_slots[ac_slot]

                if len(colliding_ues) == 0:
                    continue

                # Generate noise
                noise = np.sqrt(1/2) * (np.random.randn(frame.ac.num_channel_uses) + 1j * np.random.randn(frame.ac.num_channel_uses))

                # Get UL access channel gains
                ac_masked_channel_gains = box.get_ul_channel_gains(frame.ac.codebook[ac_slot], colliding_ues)
                ac_masked_channel_gains = np.squeeze(ac_masked_channel_gains)

                if ac_masked_channel_gains.shape == ():
                    ac_masked_channel_gains = np.array([ac_masked_channel_gains, ])

                # Obtain received signal at the AP
                buffered_access_attempts[ac_slot] = np.sqrt(snr/2) * np.sum(ac_masked_channel_gains[:, None] * messages[colliding_ues, :], axis=0) + noise

            # AP decoder
            access_dict = frame.ac.decoder(snr, num_ues, chosen_access_slots, buffered_access_attempts, messages)

            # Save results
            avg_num_successful_attempts_unicast[ap, cc, ss] = len(list_successful_ues_unicast) / num_ues
            avg_num_successful_attempts_multicast[ap, cc, ss] = len(list_successful_ues_multicast) / num_ues

##################################################
# Plot
##################################################
fig, axes = plt.subplots(3, 1)

# Go through all access policies
for ap in range(len(access_policies)):

    axes[0].plot(channel_loads, np.nanmean(avg_num_successful_attempts[ap], axis=-1), linewidth=1.5)
    axes[1].plot(channel_loads, np.nanmean(avg_num_successful_attempts_unicast[ap], axis=-1), linewidth=1.5)
    axes[2].plot(channel_loads, np.nanmean(avg_num_successful_attempts_multicast[ap], axis=-1), linewidth=1.5)

# ax.set_xlabel(r'number of access slots, $N_{\rm ac}$')
# ax.set_ylabel(r'successful attempts ratio')
#
# ax.legend(fontsize='x-small', loc='best')
#
# #ax.set_yscale('log')
#
# ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()
