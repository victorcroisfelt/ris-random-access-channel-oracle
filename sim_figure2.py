import numpy as np

from scipy import interpolate
from scipy.constants import speed_of_light

from tqdm import trange

from src.nodes import UE
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

# Noise power
noise_power_dbm = -94 # [dBm]
noise_power = 10**(noise_power_dbm/10) / 1000

########################################
# Simulation parameters
########################################

# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Place RIS
box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, size_el=size_el)

# Compute average distance
avg_distance = 2/3 * (maximum_distance**3 - minimum_distance**3) / (maximum_distance**2 - minimum_distance**2)

# Define angle of the UEs
ue_angles = np.deg2rad(np.array([30, 60]))

# Compute positions of the UEs
pos = np.zeros((2, 3))
pos[:, 0] = avg_distance * np.sin(ue_angles)
pos[:, 1] = avg_distance * np.cos(ue_angles)

# Place UEs
box.ue = UE(2, pos, None, None)

# Define a mask
mask = np.linspace(0, np.pi/2, 1000)

########################################
# Simulation
########################################

# Initialize a random access frame
frame = Frame()

# Initialize training block
frame.init_training(16, None, None, None)

# Initialize access block
frame.init_access(6, None, None, None)

# Get true DL SNR
true_snr_dl = box.get_channels(ap_tx_power, noise_power, mask, direction='dl')

# Get training sampled DL SNRs channels
tr_snr_dl = box.get_channels(ap_tx_power, noise_power, frame.tr.codebook, direction='dl')

# Get access sampled DL SNRs channels
ac_snr_dl = box.get_channels(ap_tx_power, noise_power, frame.ac.codebook, direction='dl')

# Save data
np.savez(
    'data/figure3.npz',
    mask=mask,
    tr_codebook=frame.tr.codebook,
    ac_codebook=frame.ac.codebook,
    true_snr_dl=true_snr_dl,
    tr_snr_dl=tr_snr_dl,
    ac_snr_dl=ac_snr_dl
)
