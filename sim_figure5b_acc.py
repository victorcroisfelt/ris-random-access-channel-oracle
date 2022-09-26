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
num_channel_uses_ul = 1

########################################
# Simulation
########################################

# Number of setups
num_setups = int(1e2)

# Number of noise samples
num_noisy_samples = int(1e3)

# Range of reconstruction MSE error
#range_rec_error = np.array([0, 10**(-6), 5*10**(-6), 10**(-5), 5*10**(-5), 10**(-4), 5*10**(-4), 10**(-3), 5*10**(-3), 10**(-2), 5*10**(-2), 10**(-1), 5*10**(-1), 1])
range_rec_error = np.flip(np.array([10**(-3), 10**(-2), 10**(-1)]))

# Range of number of access slots
range_ac_num_slots = np.arange(ac_min_num_slots, 201)

# Define the access policies
access_policies = ['RCARAP', 'RGSCAP', 'SMAP']

# Prepare to save accuracy
accuracy = np.zeros((len(access_policies), range_rec_error.size, range_ac_num_slots.size, num_setups, num_noisy_samples))


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

# Initialize a random access frame
frame = Frame()

# Place UEs
box.place_ue(num_setups)

# Go through all number of access slots
for asl in trange(range_ac_num_slots.size, desc="access size", unit=" as"):

    # Extract current number of access slots
    ac_num_slots = range_ac_num_slots[asl]

    # Initialize access block
    frame.init_access(ac_num_slots, num_channel_uses_ul, 0, decoding_snr)

    # True access info
    ac_true_info = box.get_channels(ue_tx_power, noise_power, frame.ac.codebook, direction='ul')

    # Generate reconstruction noise
    rec_noise = np.sqrt(1/2) * (np.random.randn(num_setups, frame.ac.num_slots, num_noisy_samples) + 1j * np.random.randn(num_setups, frame.ac.num_slots, num_noisy_samples))

    # Go through all access policies
    for ap in range(len(access_policies)):

        # Obtain true access policy
        true_chosen_ac_slots = frame.ac.access_policy(ac_true_info, access_policy=access_policies[ap], rng=np.random.RandomState(asl))

        # Go through all reconstruction errors
        for re in range(range_rec_error.size):

            # Extract current recontruction error
            rec_error = range_rec_error[re]

            # Information obtained by the UEs
            ac_info = ac_true_info[:, :, None] + np.sqrt(rec_error) * rec_noise

            # Go through all noisy samples
            for nn in range(num_noisy_samples):

                # Obtain noisy access policy
                noisy_chosen_ac_slots = frame.ac.access_policy(ac_info[:, :, nn], access_policy=access_policies[ap], rng=np.random.RandomState(asl))

                # Go through each UE
                for ue in range(num_setups):
                    intersection = (set(true_chosen_ac_slots[ue])).intersection(set(noisy_chosen_ac_slots[ue]))
                    accuracy[ap, re, asl, ue, nn] = len(set(intersection)) / len(true_chosen_ac_slots[ue])

# Save data
np.savez(
    'data/figure5b.npz',
    range_rec_error=range_rec_error,
    range_ac_num_slots=range_ac_num_slots,
    access_policies=access_policies,
    accuracy=accuracy
)
