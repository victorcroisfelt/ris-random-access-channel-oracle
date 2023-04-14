import numpy as np

from scipy import interpolate
from scipy.constants import speed_of_light

from tqdm import trange

from src.box import Box
from src.frame import Frame

import matplotlib
import matplotlib.pyplot as plt

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
size_el = wavelength / 2

# RIS size along one of the dimensions
ris_size = num_els_hor * size_el

# Distances
maximum_distance = 20
minimum_distance = (2 / wavelength) * ris_size ** 2

# DL transmit power
ap_tx_power_dbm = 20  # [dBm]
ap_tx_power = 10 ** (ap_tx_power_dbm / 10) / 1000

# UL transmit power
ue_tx_power_dbm = 10  # [dBm]
ue_tx_power = 10 ** (ue_tx_power_dbm / 10) / 1000

# Noise power
noise_power_dbm = -94  # [dBm]
noise_power = 10 ** (noise_power_dbm / 10) / 1000

########################################
# RA frame parameters
########################################

# Define MVU estimator tolerance
mvu_error_dl = mvu_error_ul = 10e-3

# Minimum SNR threshold value for decoding
decoding_snr_db = 3
decoding_snr = 10 ** (decoding_snr_db / 10)

# Compute minimum number of access slots
ac_min_num_slots = int(np.ceil(np.pi * num_els_ver * size_el / (wavelength * (2 * 1.391))))

# Number of UL channel uses
num_channel_uses_dl = int(np.ceil(1 / ((ap_tx_power / noise_power) * mvu_error_dl)))

# Number of UL channel uses
num_channel_uses_ul = 1

# Reconstruction MSE error
rec_error = 10 ** (-3)

########################################
# Simulation
########################################

# Number of setups
num_setups = int(1e5)

# Range of channel load
channel_loads = np.arange(1, 11)

# Define the access policies
access_policies = ['RCURAP', 'RCARAP', 'RGSCAP', 'SMAP']

# Define number of repetitions
num_repetitions = 1

# # Define ACK methods
# ack_methods = ['rand', 'prec', 'tdma']
# #ack_methods = ['rand']

# Prepare to store simulation results
proba_access = np.zeros((len(access_policies), channel_loads.size, num_setups))
#throughput = np.zeros((len(access_policies), channel_loads.size, num_setups))

#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

# Initialize a random access frame
frame = Frame()

# Initialize training block
frame.init_training(46, num_channel_uses_dl, 0, decoding_snr)

# Go through all channel loads
for cc in trange(channel_loads.size, desc="Channel Load", unit=" chnload"):

    # Extract current channel load
    channel_load = channel_loads[cc]

    # Generating new UEs
    range_num_ues = np.random.poisson(channel_load, (num_setups))

    # Define current number of access slots
    ac_num_slots = channel_load if channel_load >= ac_min_num_slots else ac_min_num_slots

    # Initialize access block
    frame.init_access(ac_num_slots, num_channel_uses_ul, 0, decoding_snr)

    # Go through all setups
    for ss in range(num_setups):

        # Extract current number of UEs
        num_ues = range_num_ues[ss]

        if num_ues == 0:
            proba_access[:, cc, ss] = np.nan
            throughput[:, cc, ss] = np.nan
            continue

        # Place UEs
        box.place_ue(num_ues)

        # Generate UEs messages
        ue_messages = frame.ac.messages(num_ues)

        ## DL training block

        # Generate reconstruction noise
        rec_noise = np.sqrt(1 / 2) * (
                    np.random.randn(num_ues, ac_num_slots) + 1j * np.random.randn(num_ues, ac_num_slots))

        # True access info
        ac_true_info = box.get_channels(ue_tx_power, noise_power, frame.ac.codebook, direction='ul')

        # Noisy access info
        ac_info = ac_true_info + (np.sqrt(rec_error) * rec_noise) if not np.isnan(rec_error) else ac_true_info

        ## UL access block

        # Go through all access policies
        for ap in range(len(access_policies)):

            # Extract current access policy
            access_policy = access_policies[ap]

            # Apply access policy
            ue_choices = frame.ac.access_policy(ac_info, access_policy=access_policy)

            # Get UL transmitted messages and received signals
            access_attempts, bigraph = frame.ac.ul_transmission(ac_true_info, ue_messages, ue_choices)

            # AP decoder
            access_result = frame.ac.decoder(ac_true_info, ue_messages, access_attempts, bigraph, mvu_error_ul)

            # Access number of successful UEs
            ac_num_successful_ues = len(access_result)

            if ac_num_successful_ues == 0:
                continue

            # Store simulation results
            proba_access[ap, cc, ss] = ac_num_successful_ues #/ num_ues
            #throughput[ap, cc, ss] = frame.compute_throughput(access_policy, ac_num_successful_ues)

# Save data
np.savez(
    'data/figure6.npz',
    channel_loads=channel_loads,
    access_policies=access_policies,
    proba_access=proba_access,
    #throughput=throughput
)
