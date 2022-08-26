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
# Simulation parameters
########################################

# Define SNR
snr_db = 10
snr = 10**(snr_db/10)

# Number of setups
num_setups = int(1e2)

# Define range on desired MVU estimator error tolerance
range_error_tol = 10**(-np.arange(1, 33, dtype=np.double))

# Define range on the number of training slots
range_tr_num_slots = np.arange(10, 601)

# Minimum SNR threshold value for decoding
decoding_snr = 1

########################################
# Simulation
########################################

# Initialize frame
frame = Frame()

# Test mask
test_mask = np.linspace(0, np.pi/2, 1000)

# Prepare to store simulation results
true_reconstruction_nmse = np.zeros((range_tr_num_slots.size, num_setups), dtype=np.double)
reconstruction_nmse = np.zeros((range_error_tol.size, range_tr_num_slots.size, num_setups), dtype=np.double)


#####


# Go through all error tolerances
for et in trange(range_error_tol.size, unit=" er"):

    # Extract current error tolerance
    error_tol = range_error_tol[et]

    # Go through all number of training slots
    for ts in range(range_tr_num_slots.size):

        # Extract current number of training slots
        tr_num_slots = range_tr_num_slots[ts]

        # Define number of training channel uses
        num_channel_uses = int(np.ceil(1 / (snr * error_tol)))

        # Initialize training block
        frame.init_training(tr_num_slots, num_channel_uses, 0, decoding_snr)

        # Place UEs
        box.place_ue(num_setups)

        ##################################################
        # 1) DL training block
        ##################################################

        # Get DL training channel gains
        tr_channel_gains = box.get_dl_channel_gains(frame.tr.codebook)

        # Get DL training channel gains for comparison
        true_tr_channel_gains = box.get_dl_channel_gains(test_mask)

        # Generate equivalent estimation noise
        noise = np.random.randn(num_setups, frame.tr.num_slots) + 1j * np.random.randn(num_setups, frame.tr.num_slots)
        noise *= np.sqrt(1 / (2 * frame.tr.num_channel_uses * snr))

        # Get estimated DL training channel gains
        tr_hat_channel_gains = tr_channel_gains + noise

        # Obtain reflected-angular information
        if et == 0:
            true_hat_tr_channel_gains = np.empty((num_setups, test_mask.size), dtype=np.complex_)

        hat_tr_channel_gains = np.empty((num_setups, test_mask.size), dtype=np.complex_)

        # Go through each UE
        for ue in range(num_setups):

            # Apply spline reconstruction
            interpolation_real = interpolate.interp1d(frame.tr.codebook, tr_hat_channel_gains[ue, :].real, kind='cubic')
            interpolation_imag = interpolate.interp1d(frame.tr.codebook, tr_hat_channel_gains[ue, :].imag, kind='cubic')

            # Store interpolation results
            hat_tr_channel_gains[ue] = interpolation_real(test_mask) + 1j * interpolation_imag(test_mask)

            if et == 0:

                # Apply spline reconstruction
                interpolation_real = interpolate.interp1d(frame.tr.codebook, tr_channel_gains[ue, :].real, kind='cubic')
                interpolation_imag = interpolate.interp1d(frame.tr.codebook, tr_channel_gains[ue, :].imag, kind='cubic')


                true_hat_tr_channel_gains[ue] = interpolation_real(test_mask) + 1j * interpolation_imag(test_mask)

        # Compute recontruction NMSE
        nmse = np.linalg.norm(true_tr_channel_gains - hat_tr_channel_gains, axis=-1)**2 / np.linalg.norm(true_tr_channel_gains, axis=-1)**2

        # Store it
        reconstruction_nmse[et, ts, :] = nmse

        if et == 0:

            nmse = np.linalg.norm(true_tr_channel_gains - true_hat_tr_channel_gains, axis=-1)**2 / np.linalg.norm(true_tr_channel_gains, axis=-1)**2
            true_reconstruction_nmse[ts, :] = nmse

##################################################
# Plot
##################################################
fig, ax = plt.subplots()

ax.plot(range_tr_num_slots, np.nanmean(true_reconstruction_nmse[:, :], axis=-1), linewidth=1.5, color='black')

# Go through all error tolerances
for et in range(range_error_tol.size):
    ax.plot(range_tr_num_slots, np.nanmean(reconstruction_nmse[et, :, :], axis=-1), linewidth=1.5)

ax.set_xlabel(r'number of training slots, $N_{\rm tr}$')
ax.set_ylabel(r'NMSE')

#ax.legend(fontsize='x-small', loc='best')

ax.set_yscale('log')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()
