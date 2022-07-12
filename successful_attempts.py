from environment.box import Box

import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from scipy import interpolate
from scipy.constants import speed_of_light

from randomaccessfunc import collision_resolution_slotted

from tqdm import trange

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb}')
matplotlib.rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 8})

########################################
# Preamble
########################################
seed = 42
np.random.seed(seed)

########################################
# General parameters
########################################

# Number of elements
num_els_ver = 10  # vertical
num_els_hor = 10  # horizontal

# Size of each element
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

size_el = wavelength

# RIS size along one of the dimensions
ris_size = num_els_hor * size_el

# Distances
maximum_distance = 100
minimum_distance = (2/wavelength) * ris_size**2

########################################
# Parameters: Configuration Estimation (CE) Phase
########################################

# Range of configuration estimation configs
num_ce_configs = 26

# Define SNR
snr_db = 0
snr = 10**(snr_db/10)

# Define desired MVU estimator tolerance
tolerance = 10e-12

# Compute the required number of channel uses
num_channel_uses_ce = int(np.ceil(1 / (snr * tolerance)))

########################################
# Parameters: Access Phase
########################################

# Range of access configurations
num_access_configs_range = np.arange(1, 101)
max_num_access_configs = np.max(num_access_configs_range)

# Minimum SNR threshold value
snr_th = 1

# Define the access policy
access_policy = 'strongest'
#access_policy = 'rand'
#access_policy = 'aloha'

########################################
# Simulation
########################################
print('--------------------------------------------------')
print('RIS-assisted')
print('Access policy: ' + access_policy)
print('--------------------------------------------------')

# Number of setups
num_setups = int(1e4)

# Range on the number of colliding UEs
num_ues_range = np.arange(1, 11)
max_num_ues = np.max(num_ues_range)

# Store enumeration with max number of UEs
enumeration_max_num_ues = np.arange(0, max_num_ues).astype(int)

# Prepare to store number of successful attempts
avg_num_successful_attempts = np.zeros((num_access_configs_range.size, num_ues_range.size, num_setups))
#avg_num_successful_attempts[:] = np.nan


#####


# Create a box
box = Box(maximum_distance=maximum_distance, minimum_distance=minimum_distance, rng=np.random.RandomState(seed))

# Place BS
box.place_bs(distance=minimum_distance, zenith_angle_deg=45)

# Generate UEs transmissions per slot
#transmissions_ues = np.random.poisson(lam=channel_load_range, size=(num_setups, channel_load_range.size))

# Go through all setups
for ss in trange(num_setups, desc="Simulating", unit="setup"):

    ##################################################
    # Generating new UEs
    ##################################################

    # Place UEs
    box.place_ue(int(max_num_ues))

    # Place RIS
    box.place_ris(num_els_ver=num_els_ver, num_els_hor=num_els_hor, num_ce_configs=num_ce_configs)

    ##################################################
    # Configuration estimation phase
    ##################################################

    # Obtain "full" channel gains for the configuration estimation phase
    full_channel_gains_ce = box.get_ce_channel_gains()

    # Generate equivalent noise
    noise = (np.random.randn(max_num_ues, num_ce_configs) + 1j * np.random.randn(max_num_ues, num_ce_configs))
    noise *= np.sqrt(1 / (2 * num_channel_uses_ce * snr))

    # Compute full estimated channel gains
    full_hat_channel_gains_ce = full_channel_gains_ce + noise

    # Prepare to store interpolation results
    f_real = np.empty(max_num_ues, dtype=object)
    f_imag = np.empty(max_num_ues, dtype=object)

    # Go through each UE
    for ue in enumeration_max_num_ues:

        # Apply zero-order reconstruction
        zero_order_f_real = interpolate.interp1d(box.ris.ce_codebook, full_hat_channel_gains_ce[ue, :].real, kind='cubic')
        zero_order_f_imag = interpolate.interp1d(box.ris.ce_codebook, full_hat_channel_gains_ce[ue, :].imag, kind='cubic')

        # Store interpolation results
        f_real[ue] = zero_order_f_real
        f_imag[ue] = zero_order_f_imag

    ##################################################
    # Access phase
    ##################################################

    # Prepare to save an array of dictionaries to save the UEs that chose each access slot
    access_choices = np.empty(num_access_configs_range.size, dtype=object)

    # Go through all number of access configurations
    for cc, num_access_configs in enumerate(num_access_configs_range):

        # Get access configurations
        box.ris.set_access_codebook(num_access_configs)
        access_configs = box.ris.get_access_codebook()

        # Prepare to store reconstructed channel gains
        full_recon_channel_gains_access = np.empty((max_num_ues, num_access_configs), dtype=np.complex_)

        # Go through all UEs
        for ue in enumeration_max_num_ues:

            # Store reconstructed channel gains
            full_recon_channel_gains_access[ue, :] = f_real[ue](access_configs) + 1j * f_imag[ue](access_configs)

        # Calculate absolute values
        full_recon_absolutes_access = np.abs(full_recon_channel_gains_access)

        # Prepare to save chosen access slots
        access_choices[cc] = {key: [] for key in range(num_access_configs)}

        # Choose access slots
        if access_policy == 'strongest':

            # Store UE choices
            ue_choices = np.argmax(full_recon_absolutes_access, axis=-1)

            # Go through all UEs
            for ue in enumeration_max_num_ues:

                # Go through all access slots chosen by that UE
                try:
                    for access in ue_choices[ue]:
                        access_choices[cc][access].append(ue)
                except:
                    access_choices[cc][ue_choices[ue]].append(ue)

    ##################################################
    # Collision resolutions
    ##################################################

    # Go through all collision sizes
    for nn, num_ues in enumerate(num_ues_range):

        # Randomly selects UEs to collide
        colliding_ues = np.random.choice(enumeration_max_num_ues, num_ues, replace=False)

        # Go through all number of access configurations
        for cc, num_access_configs in enumerate(num_access_configs_range):

            # Go through all access slots
            for access in range(num_access_configs):

                # Get UEs that chosen this slot
                ues = access_choices[cc][access]

                # Delete non-colliding UEs
                true_ues = set.intersection(set(ues), set(colliding_ues))

                if len(true_ues) == 1:
                    avg_num_successful_attempts[cc, nn, ss] += 1

    ##################################################
    # ACK
    ##################################################
    # TODO:


##################################################
# Plot
##################################################
fig, ax = plt.subplots()

# Go through all collision sizes
for nn, num_ues in enumerate(num_ues_range):
    ax.plot(num_access_configs_range, avg_num_successful_attempts[:, nn].mean(axis=-1) / num_ues, linewidth=1.5, label='$K=' + str(num_ues) + '$')

ax.set_xlabel(r'number of access slots, $N_{\rm ac}$')
ax.set_ylabel(r'successful attempts ratio')

ax.legend(fontsize='x-small', loc='best')

#ax.set_yscale('log')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()

# # Save data
# np.savez('data/ris_' + access_policy + '_N' + str(Nx * Nz) + '_slot_based.npz',
#          num_configs_range=num_configs_range,
#          channel_load_range=channel_load_range,
#          avg_num_successful_attempts=avg_num_successful_attempts
#          )

#
#     # Current number of UEs
#     num_ues = num_ues_range[ii]
#
#     # Enumerate new UEs
#     enumeration_ues = np.arange(0, num_ues)
#
#     # Mask out
#     channel_gains_dl, channel_gains_ul = complete_channel_gains_dl


#

#

#

#
#
#
#         # Prepare to store UE channel gains
#         ue_channel_gains = {}
#
#         # Prepare to store UE policies
#         ue_policies = {}
#
#
#
#
#
#
#
#         # Compute received signals
#         received_signals = np.sqrt(snr) * channel_gains_dl[ue, ce, None, None] * reference_signal[:, :] + noise[:, :]
#
#
#
#
#         ##################################################
#         # Access phase
#         ##################################################
#

#
#         elif access_policy == 'rand':
#
#             # Prepare to save access frames probabilities for each UE
#             access_frame_probabilities = np.zeros((num_access_slots, total_num_new_ues))
#
#             # Compute vector of probabilities
#             access_frame_probabilities[:] = configs_strength / configs_strength.sum(axis=0)[np.newaxis, :]
#
#             # Store choices
#             for new_ue in enumeration_total_new_ues:
#                 ue_policies[new_ue] = list(access_frame_probabilities[:, new_ue])
#
#
#         # Prepare to store UE access attempts
#         ue_access = {}
#
#         # Go through all access slots
#         for aa in enumeration_access_slots:
#
#             # Extract current number of new UEs
#             current_num_new_ues = num_new_ues[aa]
#
#             if current_num_new_ues == 0:
#                 continue
#
#             # UEs that became active in this slot
#             enumeration_new_ues = enumeration_total_new_ues[num_new_ues[:aa].sum():num_new_ues[:(aa+1)].sum()]
#
#             # Go through all active UEs
#             for ue in enumeration_new_ues:
#
#                 # Choose access frames
#                 if access_policy == 'strongest':
#
#                     if ue_policies[ue] >= aa:
#                         ue_access[ue] = ue_policies[ue]
#
#                 elif access_policy == 'rand':
#
#                     # Prepare to save selected access frames
#                     selected = np.zeros(num_access_slots - aa).astype(int)
#
#                     # Flipping coins
#                     tosses = np.random.rand(num_access_slots - aa)
#
#                     # Selected access frames
#                     selected[tosses <= ue_policies[ue][aa:]] = True
#
#                     # Store list of choices
#                     choices = [access_slot for access_slot in enumeration_access_slots[aa:] if selected[access_slot - aa] == 1]
#
#                     if len(choices) >= 1:
#
#                         # Store choices
#                         ue_access[ue] = choices
#
#                     del choices
#
#                 elif access_policy == 'aloha':
#
#                     # Prepare to save selected access frames
#                     selected = np.zeros(num_access_slots - aa).astype(int)
#
#                     # Flipping coins
#                     tosses = np.random.rand(num_access_slots - aa)
#
#                     # Selected access frames
#                     selected[tosses <= 0.5] = True
#
#                     # Store list of choices
#                     choices = [access_slot for access_slot in enumeration_access_slots[aa:] if selected[access_slot - aa] == 1]
#
#                     if len(choices) >= 1:
#
#                         # Store choices
#                         ue_access[ue] = choices
#
#                     del choices
#
#         ##################################################
#         ## Buffered signal
#         ##################################################
#
#         if len(ue_access.keys()) == 0:
#             continue
#
#         # Generate noise vector at the BS
#         noise_bs = np.sqrt(noise_power / 2) * (np.random.randn(num_access_slots) + 1j * np.random.randn(num_access_slots))
#
#         # Store buffered UL received signal responses as a dictionary
#         buffered_access_attempts = {access_slot: {} for access_slot in enumeration_access_slots}
#
#         # Go through all access slots
#         for aa in enumeration_access_slots:
#
#             # Create another dictionary
#             buffered_access_attempts[aa] = {}
#
#             # Go through all active UEs
#             for ue in ue_access.keys():
#
#                 if aa in ue_access[ue]:
#                     buffered_access_attempts[aa][ue] = np.sqrt(box.ue.max_pow) * channel_gains_ul[aa, ue] * np.sqrt(1 / 2) * (1 + 1j)
#
#             # Add noise
#             buffered_access_attempts[aa]['noise'] = noise_bs[aa]
#
#         ##################################################
#         ## Collision resolution strategy
#         ##################################################
#
#         # Get number of successful attempts
#         num_successful_attempts = collision_resolution_slotted(ue_access, buffered_access_attempts, gamma_th)
#
#         # Average with respect to how much was sent
#         avg_num_successful_attempts[cc, ii, ss] = num_successful_attempts/len(ue_access.keys())