import numpy as np

from scipy import interpolate
from scipy.constants import speed_of_light

from tqdm import trange

from src.box import Box
from src.frame import Frame

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

# APs transmit power
ap_tx_power_dbm = 20 # [dBm]
ap_tx_power = 10**(ap_tx_power_dbm/10) / 1000

# Noise power
noise_power_dbm = -94 # [dBm]
noise_power = 10**(noise_power_dbm/10) / 1000

########################################
# Simulation parameters
########################################

# Number of setups
num_setups = int(1e4)

# Define range on desired MVU estimator error tolerance
range_mvu_error_dl = np.concatenate((np.array([np.nan, ]), 10**(-np.arange(1, 4, dtype=np.double))))

# Define range on the number of training slots
range_tr_num_slots = np.arange(10, 201)

########################################
# Simulation
########################################

# Test mask
test_mask = np.linspace(0, np.pi/2, 100)

# Prepare to store simulation results
reconstruction_nmse = np.zeros((range_mvu_error_dl.size, range_tr_num_slots.size, num_setups), dtype=np.double)


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

# Place UEs
box.place_ue(num_setups)

# Initialize frame
frame = Frame()

# Go through all number of training slots
for ts in trange(range_tr_num_slots.size, unit=' ts', colour='green'):

    # Extract current number of training slots
    tr_num_slots = range_tr_num_slots[ts]

    # Initialize training block
    frame.init_training(tr_num_slots, None, 0, decoding_snr=None)

    # Get DL training channels
    tr_channels = box.get_channels(ap_tx_power, noise_power, frame.tr.codebook, direction='dl')

    # Get true DL training channels for comparison
    true_tr_channels = box.get_channels(ap_tx_power, noise_power, test_mask, direction='dl')

    # Go through all error tolerances
    for et in range(range_mvu_error_dl.size):

        # Extract current error tolerance
        mvu_error_dl = range_mvu_error_dl[et]

        if np.isnan(mvu_error_dl):

            # Noiseless case
            tr_hat_channels = tr_channels

        else:

            # Generate equivalent estimation noise
            noise = np.random.randn(num_setups, frame.tr.num_slots) + 1j * np.random.randn(num_setups, frame.tr.num_slots)
            noise *= np.sqrt(1/2) * np.sqrt(mvu_error_dl)

            # Get estimated DL training channels
            tr_hat_channels = tr_channels + noise

        # Obtain reflected-angular information
        hat_tr_channels = np.empty((num_setups, test_mask.size), dtype=np.complex_)

        # Go through each UE
        for ue in range(num_setups):

            # Apply spline reconstruction
            interpolation_real = interpolate.interp1d(frame.tr.codebook, tr_hat_channels[ue, :].real, kind='cubic')
            interpolation_imag = interpolate.interp1d(frame.tr.codebook, tr_hat_channels[ue, :].imag, kind='cubic')

            # Store interpolation results
            hat_tr_channels[ue] = interpolation_real(test_mask) + 1j * interpolation_imag(test_mask)

        # Compute recontruction NMSE
        nmse = np.linalg.norm(true_tr_channels - hat_tr_channels, axis=-1)**2 / np.linalg.norm(true_tr_channels, axis=-1)**2

        # Store it
        reconstruction_nmse[et, ts, :] = nmse

# Save data
np.savez(
    'data/evaluating_dl_training_block.npz',
    reconstruction_nmse=reconstruction_nmse,
    range_tr_num_slots=range_tr_num_slots,
    range_mvu_error_dl=range_mvu_error_dl
)
