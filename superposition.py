import numpy as np
from scipy.constants import speed_of_light

def channel_model(gain_bs, gain_ue, dist_bs, dist_ue, ang_bs, ang_ue, config_ris, size_el, num_els_x, num_els_z):
    """Evaluates channel model

    Returns
    -------
    channel_gains_dl : ndarray of shape (num_configs, num_ues)
        Downlink channel gain between the BS and each UE for each RIS configuration.

    channel_gains_ul : ndarray of shape (num_configs, num_ues)
        Uplink channel gain between the BS and each UE for each RIS configuration.

    """
    # COmpute wavenumber
    wavelength = speed_of_light / carrier_frequency
    wavenumber = 2 * np.pi / wavelength

    # Compute constant term
    num = gain_bs * gain_ue * (size_el * size_el)**2
    den = (4 * np.pi * dist_bs * dist_ue)**2

    const = num/den

    # Compute DL pathloss component of shape (num_ues, )
    pathloss_dl = const * np.cos(ang_bs)**2

    # Compute UL pathloss component of shape (num_ues, )
    pathloss_ul = const * np.cos(ang_ue)**2

    # Compute constant phase component of shape (num_ues, )
    distances_sum = dist_bs + dist_ue
    disagreement = (np.sin(ang_bs) - np.sin(ang_ue)) * ((num_els_x + 1) / 2) * size_el

    phi = - wavenumber * (distances_sum - disagreement)

    # Compute array factor of shape (num_configs, num_ues)
    enumeration_num_els_x = np.arange(1, num_els_x + 1)
    sine_differences = np.sin(ang_ue) - np.sin(config_ris)

    argument = wavenumber * sine_differences * enumeration_num_els_x * size_el

    array_factor_dl = num_els_z * np.sum(np.exp(+1j * argument))
    array_factor_ul = array_factor_dl.conj()

    # Compute channel gains of shape (num_configs, num_ues)
    channel_gains_dl = np.sqrt(pathloss_dl) * np.exp(+1j * phi) * array_factor_dl
    channel_gains_ul = np.sqrt(pathloss_ul) * np.exp(-1j * phi) * array_factor_ul

    return channel_gains_dl, channel_gains_ul

########################################
# Preamble
########################################
seed = 42
np.random.seed(seed)

########################################
# General parameters
########################################

# Number of elements
Nz = 10  # vertical
Nx = 10  # horizontal

# Size of each element
carrier_frequency = 3e9
wavelength = speed_of_light / carrier_frequency

el_size = wavelength

# RIS size along one of the dimensions
ris_size = Nx * el_size

# Distances
maximum_distance = 100
minimum_distance = (2/wavelength) * ris_size**2

# Noise power
noise_power = 10 ** (-94.0 / 10)  # mW

########################################
# Simulation parameters
########################################

gain_bs = gain_ue = 10**(5/10)

dist_bs = 25
dist_ue = 25

ang_bs = np.deg2rad(45)
ang_ue = np.deg2rad(45)

size_el = wavelength

num_els_x = num_els_z = 10


#####


config_ris = np.deg2rad(15)

cg_dl_1, cg_ul_1 = channel_model(gain_bs, gain_ue, dist_bs, dist_ue, ang_bs, ang_ue, config_ris, size_el, num_els_x, num_els_z)

config_ris = np.deg2rad(30)

cg_dl_2, cg_ul_2 = channel_model(gain_bs, gain_ue, dist_bs, dist_ue, ang_bs, ang_ue, config_ris, size_el, num_els_x, num_els_z)

config_ris = np.deg2rad(45)

cg_dl_3, cg_ul_3 = channel_model(gain_bs, gain_ue, dist_bs, dist_ue, ang_bs, ang_ue, config_ris, size_el, num_els_x, num_els_z)