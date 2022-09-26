import numpy as np

from scipy import interpolate
from scipy.constants import speed_of_light

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
# Simulation
########################################

# Number of realizations
num_realizations = int(1e3)

# Place UEs
box.place_ue(num_realizations)

# Evaluate the best possible channel responses (reflected angle is equal to the
# UE angle)
dl_channels = box.get_channels(1, 1, box.ue.angles, direction='dl')

# Flat the array
dl_channels_flattened = dl_channels.flatten()

# Compute the channel gains
dl_channel_gains = np.abs(dl_channels_flattened)**2

# Sort them
dl_channel_gains_sorted = np.sort(dl_channel_gains)

# Compute in dB
dl_channel_gains_sorted_db = 10 + 94 + 10 * np.log10(dl_channel_gains_sorted)

# Get CDF
dl_channel_gains_cdf = np.linspace(0, 1, dl_channel_gains_sorted.size)

print('----- Summary Stats ------')
print('\t Min.:', dl_channel_gains_sorted_db[0])
print('\t Max.:', dl_channel_gains_sorted_db[-1])
print('\t Avg.:', dl_channel_gains_sorted_db.mean())
print('\t Median:', np.median(dl_channel_gains_sorted_db))
print('\t 1st Quartile:', np.percentile(dl_channel_gains_sorted_db, q=25))
print('\t 3nd Quartile:', np.percentile(dl_channel_gains_sorted_db, q=75))

##################################################
# Plot
##################################################
fig, ax = plt.subplots()

ax.plot(dl_channel_gains_sorted_db, dl_channel_gains_cdf, linewidth=1.5, color='black')

ax.set_ylabel(r'CDF')
ax.set_xlabel(r'DL channel gains [dB]')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()
