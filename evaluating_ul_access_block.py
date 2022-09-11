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
seed = 0
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
decoding_snr_db = 0
decoding_snr = 10**(decoding_snr_db/10)

# Compute minimum number of access slots
ac_min_num_slots = int(np.ceil(np.pi * num_els_ver * size_el / (wavelength * (2 * 1.391))))

# Compute UL transmit power
#avg_chn_gain = ((10**(5/10))**2 / ((4 * np.pi) ** 2)) * (size_el**2/minimum_distance)**2 * ((np.log(maximum_distance) - np.log(minimum_distance))/(maximum_distance**2 - minimum_distance**2))
#ue_tx_power = (noise_power / avg_chn_gain / ((num_els_hor * num_els_ver)**2) / 0.5) * decoding_snr

# Number of UL channel uses
num_channel_uses_dl = int(np.ceil(1 / ((ap_tx_power/noise_power) * mvu_error_dl)))

# Number of UL channel uses
num_channel_uses_ul = int(np.ceil(1 / ((ue_tx_power/noise_power) * mvu_error_ul)))

# Reconstruction MSE error
rec_error = 10**(-3)

########################################
# Simulation
########################################

# Number of setups
num_setups = int(1e4)

# Range of channel load
channel_loads = np.arange(1, 11)

# Define the access policies
access_policies = ['RCURAP', 'RCARAP', 'RGSCAP', 'SMAP']

# Range of number of access slots
range_ac_num_slots = np.arange(ac_min_num_slots, 11, dtype=int)

# Prepare to store simulation results
successful_attempts_ratio = np.zeros((len(access_policies), range_ac_num_slots.size, channel_loads.size, num_setups))
throughput = np.zeros((len(access_policies), range_ac_num_slots.size, channel_loads.size, num_setups))

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

    # Go through all setups
    for ss in range(num_setups):

        # Extract current number of UEs
        num_ues = range_num_ues[ss]

        if num_ues == 0:
            successful_attempts_ratio[:, :, cc, ss] = np.nan
            throughput[:, :, cc, ss] = np.nan
            continue

        # Place UEs
        box.place_ue(num_ues)

        # Generate reconstruction noise
        rec_noise = np.sqrt(1/2) * (np.random.randn(range_ac_num_slots.max()) + 1j * np.random.randn(range_ac_num_slots.max()))

        # Go through all number of access slots
        for ns in range(range_ac_num_slots.size):

            # Extract current number of access slots
            ac_num_slots = range_ac_num_slots[ns]

            # Initialize access block
            frame.init_access(ac_num_slots, num_channel_uses_ul, 0, decoding_snr)

            # Generate UEs messages
            if ns == 0:
                ue_messages = frame.ac.messages(num_ues)

            # True access info
            ac_info = box.get_channels(ue_tx_power, noise_power, frame.ac.codebook, direction='ul')

            if not np.isnan(rec_error):

                # Noisy access info
                ac_info += np.sqrt(rec_error) * rec_noise[:ac_num_slots]

            # Go through all access policies
            for ap in range(len(access_policies)):

                # Reset seed
                np.random.seed(ns)

                # Extract current access policy
                access_policy = access_policies[ap]

                # Apply access policy
                chosen_access_slots = frame.ac.access_policy(ac_info, access_policy=access_policy)

                # Get UL access channels
                channels_ul = box.get_channels(ue_tx_power, noise_power, frame.ac.codebook, direction='ul')

                # Get UL transmitted messages and received signals
                buffered_access_attempts = frame.ac.ul_transmission(num_ues, channels_ul, ue_messages, chosen_access_slots)

                # AP decoder
                access_dict = frame.ac.decoder(num_ues, chosen_access_slots, ue_messages, buffered_access_attempts)

                # Extract successful access slots
                successful_ac_slots = list(access_dict.keys())

                # Extract successful UEs
                successful_ac_ues = [ue for ac in successful_ac_slots for ue in access_dict[ac]]

                if len(successful_ac_ues) > num_ues:
                    raise NameError('number of successful users is larger than the actual number of users')

                # Compute number of successful attempts
                current_num_successful_attempts = len(successful_ac_ues)

                # Compute throughput
                current_throughput = frame.compute_throughput(access_policy, current_num_successful_attempts)

                # Store simulation results
                successful_attempts_ratio[ap, ns, cc, ss] = current_num_successful_attempts / num_ues
                throughput[ap, ns, cc, ss] = current_throughput

# Save data
np.savez(
    'data/evaluating_ul_access_block.npz',
    successful_attempts_ratio=successful_attempts_ratio,
    channel_loads=channel_loads,
    range_ac_num_slots=range_ac_num_slots,
    access_policies=access_policies
)

fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)

axes = axes.flatten()

for ap in range(len(access_policies)):

    # Go through all channel loads
    for cc, channel_load in enumerate(channel_loads):

        axes[ap].plot(range_ac_num_slots, np.nanmean(successful_attempts_ratio[ap, :, cc, :], axis=-1))


fig, axes = plt.subplots(ncols=2, nrows=2, sharex=True, sharey=True)

axes = axes.flatten()

for ap in range(len(access_policies)):

    # Go through all channel loads
    for cc, channel_load in enumerate(channel_loads):

        axes[ap].plot(range_ac_num_slots, np.nanmean(throughput[ap, :, cc, :], axis=-1))

plt.show()
