import numpy as np

from scipy import interpolate
from scipy.constants import speed_of_light

from tqdm import trange

from src.box import Box
from src.frame import Frame

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

# APs transmit power
ap_tx_power_dbm = 20 # [dBm]
ap_tx_power = 10**(ap_tx_power_dbm/10) / 1000

# UEs transmit power
ue_tx_power_dbm = 10 # [dBm]
ue_tx_power = 10**(ue_tx_power_dbm/10) / 1000

# Noise power
noise_power_dbm = -94 # [dBm]
noise_power = 10**(noise_power_dbm/10) / 1000

########################################
# RA frame parameters
########################################

# Define desired MVU estimator tolerance
mvu_error_ul = 10e-3

# Minimum SNR threshold value for decoding
decoding_snr_db = 0
decoding_snr = 10**(decoding_snr_db/10)

# Compute minimum number of access slots
ac_num_slots = int(np.ceil(np.pi * num_els_ver * size_el / (wavelength * (2 * 1.391))))

# Compute minimum number of training slots
tr_num_slots = 46

# Compute minimum UL transmit power
avg_chn_gain = ((10**(5/10))**2 / ((4 * np.pi) ** 2)) * (size_el**2/minimum_distance)**2 * ((np.log(maximum_distance) - np.log(minimum_distance))/(maximum_distance**2 - minimum_distance**2))
ue_tx_power = (noise_power / avg_chn_gain / ((num_els_hor * num_els_ver)**2) / 0.5) * decoding_snr

# Number of UL channel uses
num_channel_uses_ul = int(np.ceil(1 / ((ue_tx_power/noise_power) * mvu_error_ul)))

########################################
# Simulation
########################################

# Number of setups
num_setups = int(1e4)

# Number of packets to be transmited per UE
num_packets = 1

# Range of channel load
channel_loads = np.arange(1, 11)

# Range of DL MVU error
range_mvu_error_dl = np.concatenate((np.array([np.nan, ]), 10**(-np.arange(1, 4, dtype=np.double))))

# Define the access policies
access_policies = ['RCURAP', 'RCARAP', 'RGSCAP', 'SMAP']

# Prepare to store number of successful attempts
num_successful_attempts = np.zeros((len(access_policies), range_mvu_error_dl.size, channel_loads.size, num_setups))


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

# Initialize a random access frame
frame = Frame()

# Go through all channel loads
for cc in trange(channel_loads.size, desc="Channel Load", unit=" chnload"):

    # Extract current channel load
    channel_load = channel_loads[cc]

    if channel_load <= ac_num_slots:

        # Initialize access block
        frame.init_access(ac_num_slots, num_channel_uses_ul, 0, decoding_snr, 1/2, num_els_hor)

    else:

        # Initialize access block
        frame.init_access(channel_load, num_channel_uses_ul, 0, decoding_snr, 1/2, num_els_hor)
        breakpoint()
    # Generating new UEs
    range_num_ues = np.random.poisson(channel_load, (num_setups))

    # Go through all setups
    for ss in range(num_setups):

        # Extract current number of UEs
        num_ues = range_num_ues[ss]

        if num_ues == 0:

            num_successful_attempts[ap, re, cc, ss] = np.nan
            continue

        # Enumerate UEs
        enumeration_ues = np.arange(0, num_ues)

        # Place UEs
        box.place_ue(num_ues)

        # Go through all DL MVU errors
        for re in range(range_mvu_error_dl.size):

            # Extract current DL MVU error
            mvu_error_dl = range_mvu_error_dl[re]

            # Initialize training block
            frame.init_training(tr_num_slots, None, 0, decoding_snr=None)

            # Get DL training channels
            tr_channels = box.get_channels(ap_tx_power, noise_power, frame.tr.codebook, direction='dl')

            # Get estimated DL training channels
            if np.isnan(mvu_error_dl):
                tr_hat_channels = tr_channels

            else:

                # Generate equivalent estimation noise
                noise = np.random.randn(num_ues, frame.tr.num_slots) + 1j * np.random.randn(num_ues, frame.tr.num_slots)
                noise *= np.sqrt(1/2) * np.sqrt(mvu_error_dl)

                tr_hat_channels = tr_channels + noise

            # Obtain reflected-angular information
            ac_info = np.empty((num_ues, frame.ac.num_slots), dtype=np.complex_)

            # Go through each UE
            for ue in range(num_ues):

                # Apply spline reconstruction
                interpolation_real = interpolate.interp1d(frame.tr.codebook, tr_hat_channels[ue, :].real, kind='cubic')
                interpolation_imag = interpolate.interp1d(frame.tr.codebook, tr_hat_channels[ue, :].imag, kind='cubic')

                # Store interpolation results
                ac_info[ue] = interpolation_real(frame.ac.codebook) + 1j * interpolation_imag(frame.ac.codebook)

            # Go through all access policies
            for ap in range(len(access_policies)):

                # Ensure that randomness is the same for each access policy
                seed = 42
                np.random.seed(seed)

                # Apply access policy
                chosen_access_slots = frame.ac.access_policy(ac_info, num_packets=num_packets, policy=access_policies[ap])

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

                    # Get UL access channels
                    ac_masked_channels = box.get_channels(ue_tx_power, noise_power, frame.ac.codebook[ac_slot], direction='ul', mask=colliding_ues)
                    ac_masked_channels = np.squeeze(ac_masked_channels)

                    if ac_masked_channels.shape == ():
                        ac_masked_channels = np.array([ac_masked_channels, ])

                    # Obtain received signal at the AP
                    buffered_access_attempts[ac_slot] = np.sum(ac_masked_channels[:, None] * messages[colliding_ues, :], axis=0) + noise

                # AP decoder
                access_dict = frame.ac.decoder(num_ues, chosen_access_slots, buffered_access_attempts, messages)

                # Extract successful access slots
                successful_ac_slots = list(access_dict.keys())

                # Extract successful UEs
                successful_ac_ues = [ue for ac in successful_ac_slots for ue in access_dict[ac]]

                if len(successful_ac_ues) > num_ues:
                    raise NameError('number of successful users is larger than the actual number of users')

                # Save itermediate result
                num_successful_attempts[ap, re, cc, ss] = len(successful_ac_ues) / num_ues

# Save data
np.savez(
    'data/evaluating_dl_training_block_spline.npz',
    num_successful_attempts=num_successful_attempts,
    channel_loads=channel_loads,
    range_mvu_error_dl=range_mvu_error_dl,
    access_policies=access_policies
)
