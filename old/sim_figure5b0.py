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

# Define UL MVU estimator tolerance
mvu_error_ul = 10e-3

# Minimum SNR threshold value for decoding
decoding_snr_db = 3
decoding_snr = 10**(decoding_snr_db/10)

# Compute minimum number of access slots
ac_min_num_slots = int(np.ceil(np.pi * num_els_ver * size_el / (wavelength * (2 * 1.391))))

# Compute UL transmit power
# avg_chn_gain = ((10**(5/10))**2 / ((4 * np.pi) ** 2)) * (size_el**2/minimum_distance)**2 * ((np.log(maximum_distance) - np.log(minimum_distance))/(maximum_distance**2 - minimum_distance**2))
# ue_tx_power = (noise_power / avg_chn_gain / ((num_els_hor * num_els_ver)**2) / 0.5) * decoding_snr

# Number of UL channel uses
num_channel_uses_ul = int(np.ceil(1 / ((ue_tx_power/noise_power) * mvu_error_ul)))

########################################
# Simulation
########################################

# Number of setups
num_setups = int(1e4)

# Range of channel load
channel_loads = np.arange(1, 11)

# Range of reconstruction MSE error
range_rec_error = np.array([0, 10**(-6), 5*10**(-6), 10**(-5), 5*10**(-5), 10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3), 10**(-2), 5*10**(-2), 10**(-1), 5*10**(-1), 1])

# Define the access policies
access_policies = ['RCURAP', 'RCARAP', 'RGSCAP', 'SMAP']

# Prepare to store number of successful attempts
num_successful_attempts = np.zeros((len(access_policies), range_rec_error.size, channel_loads.size, num_setups))


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

    # Generating new UEs
    range_num_ues = np.random.poisson(channel_load, (num_setups))

    # Initialize access block
    if channel_load <= ac_min_num_slots:
        frame.init_access(ac_min_num_slots, num_channel_uses_ul, 0, decoding_snr)
    else:
        frame.init_access(channel_load, num_channel_uses_ul, 0, decoding_snr)

    # Go through all setups
    for ss in range(num_setups):

        # Extract current number of UEs
        num_ues = range_num_ues[ss]

        if num_ues == 0:
            num_successful_attempts[:, :, cc, ss] = np.nan
            continue

        # Place UEs
        box.place_ue(num_ues)

        # Generate UEs messages
        ue_messages = frame.ac.messages(num_ues)

        # True access info
        ac_true_info = box.get_channels(ue_tx_power, noise_power, frame.ac.codebook, direction='ul')

        # Generate reconstruction noise
        rec_noise = np.sqrt(1/2) * (np.random.randn(frame.ac.num_slots) + 1j * np.random.randn(frame.ac.num_slots))

        # Go through all reconstruction errors
        for re in range(range_rec_error.size):

            # Extract current recontruction error
            rec_error = range_rec_error[re]

            # Information obtained by the UEs
            ac_info = ac_true_info + np.sqrt(rec_error) * rec_noise

            # Go through all access policies
            for ap in range(len(access_policies)):

                seed = 42
                np.random.seed(seed)

                # Apply access policy
                chosen_access_slots = frame.ac.access_policy(ac_info, access_policy=access_policies[ap])

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

                # Store simulation result
                num_successful_attempts[ap, re, cc, ss] = len(successful_ac_ues) / num_ues

# Save data
np.savez(
    'data/figure5b.npz',
    num_successful_attempts=num_successful_attempts,
    channel_loads=channel_loads,
    range_rec_error=range_rec_error,
    access_policies=access_policies
)
