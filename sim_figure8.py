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
size_el = wavelength/2

# RIS size along one of the dimensions
ris_size = num_els_hor * size_el

# Distances
maximum_distance = 100
minimum_distance = (2/wavelength) * ris_size**2

# DL transmit power
ap_tx_power_dbm = 20 # [dBm]
ap_tx_power = 10**(ap_tx_power_dbm/10) / 1000

# UL transmit power
ue_tx_power_dbm = 10 # [dBm]
ue_tx_power = 10**(ue_tx_power_dbm/10) / 1000

# Noise power
noise_power_dbm = -94 # [dBm]
noise_power = 10**(noise_power_dbm/10) / 1000

########################################
# RA frame parameters
########################################

# Define MVU estimator tolerance
mvu_error_dl = mvu_error_ul = 10e-3

# Minimum SNR threshold value for decoding
decoding_snr_db = 3
decoding_snr = 10**(decoding_snr_db/10)

# Compute minimum number of access slots
ac_min_num_slots = int(np.ceil(np.pi * num_els_ver * size_el / (wavelength * (2 * 1.391))))

# # Compute UL transmit power
# avg_chn_gain = ((10**(5/10))**2 / ((4 * np.pi) ** 2)) * (size_el**2/minimum_distance)**2 * ((np.log(maximum_distance) - np.log(minimum_distance))/(maximum_distance**2 - minimum_distance**2))
# ue_tx_power = (noise_power / avg_chn_gain / ((num_els_hor * num_els_ver)**2) / 0.5) * decoding_snr

# Number of UL channel uses
num_channel_uses_dl = int(np.ceil(1 / ((ap_tx_power/noise_power) * mvu_error_dl)))

# Number of UL channel uses
num_channel_uses_ul = 1

# Reconstruction MSE error
rec_error = 10**(-3)

########################################
# Simulation
########################################

# Number of setups
num_setups = int(1e5)

# Range of channel load
channel_loads = np.arange(1, 151)

# Range of switch time
switch_times = np.array([0, 1, 5])

# Define the access policies
access_policies = ['RCURAP', 'RCARAP', 'RGSCAP', 'SMAP']

# Define number of repetitions
num_repetitions = 1

# Define ACK methods
ack_methods = ['prec', 'tdma']

# Prepare to store number of successful attempts
proba_access = np.zeros((len(ack_methods) + 1, len(access_policies), channel_loads.size, num_setups))
throughput = np.zeros((len(switch_times), len(ack_methods) + 1, len(access_policies), channel_loads.size, num_setups))

#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

# Initialize a random access frame
frame = Frame()

# Go through all switch times


# Initialize training block
frame.init_training(46, num_channel_uses_dl, 0, decoding_snr)


# Go through all channel loads
for cc in trange(channel_loads.size, desc="Channel Load", unit=" chnload", colour="green"):

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
            proba_access[:, :, cc, ss] = np.nan
            throughput[:, :, :, cc, ss] = np.nan
            continue

        # Place UEs
        box.place_ue(num_ues)

        # Generate UEs messages
        ue_messages = frame.ac.messages(num_ues)


        ## DL training block


        # Generate reconstruction noise
        rec_noise = np.sqrt(1/2) * (np.random.randn(num_ues, ac_num_slots) + 1j * np.random.randn(num_ues, ac_num_slots))

        # True access info
        ac_true_info = box.get_channels(ue_tx_power, noise_power, frame.ac.codebook, direction='ul')

        # Noisy access info
        ac_info = ac_true_info + np.sqrt(rec_error) * rec_noise if not np.isnan(rec_error) else ac_true_info


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
            proba_access[0, ap, cc, ss] = ac_num_successful_ues / num_ues


            ## DL ackowledgment block


            # Initialize ACK block
            frame.init_ack(ac_num_successful_ues, num_channel_uses_dl, 0, decoding_snr)

            # Get successful UEs
            ac_successful_ues = np.array(list(access_result.keys()))

            # Get detected direction
            mask = np.squeeze(list(access_result.values()))
            detected_directions = frame.ac.codebook[mask]

            if type(detected_directions) is np.float_:
                detected_directions = np.array([detected_directions,])

            temp = []

            # Go through all ACK methods
            for aa, ack_method in enumerate(ack_methods):

                # Set codebook
                frame.ack.set_codebook(detected_directions, ack_method=ack_method)

                # Generate DL channels
                dl_channels = box.get_channels(ap_tx_power, noise_power, frame.ack.codebook, direction='dl', mask=ac_successful_ues)

                # Get number of ACK success
                ack_num_successful_ues = frame.ack.dl_transmission(dl_channels)

                # Store simulation results
                proba_access[aa + 1, ap, cc, ss] = ack_num_successful_ues / num_ues
                temp.append(ack_num_successful_ues)

            for st, switch_time in enumerate(switch_times):
                throughput[st, 0, ap, cc, ss] = frame.compute_throughput(access_policy, ac_num_successful_ues, switch_time=switch_time)
                throughput[st, 1, ap, cc, ss] = frame.compute_throughput(access_policy, temp[0], ack_method='prec', switch_time=switch_time)
                throughput[st, 2, ap, cc, ss] = frame.compute_throughput(access_policy, temp[1], ack_method='tdma', switch_time=switch_time)

            del temp

# Save data
np.savez(
    'data/figure8.npz',
    channel_loads=channel_loads,
    access_policies=access_policies,
    ack_methods=ack_methods,
    switch_times=switch_times,
    proba_access=proba_access,
    throughput=throughput
)
