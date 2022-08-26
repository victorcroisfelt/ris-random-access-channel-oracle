import matplotlib
import matplotlib.pyplot as plt

import numpy as np

from scipy.constants import speed_of_light
import scipy.integrate as integrate

from tqdm import trange

 # LaTeX type definitions
plt.rcParams.update({
    'text.usetex': True,
    'font.family': "serif",
    'font.serif': ["Times"]
    })

#plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 14})
#plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,amsfonts,amsthm,mathtools,cuted,bbold} \usepackage[cmintegrals]{newtxmath}')

########################################
# Private functions
########################################

def compute_expected_throughput(
    channel_load,
    ac_range_num_slots,
    ac_num_channel_uses,
    tr_duration,
    ack_duration,
    ):

    num = channel_load**2 / ac_range_num_slots * (1 - 1/ac_range_num_slots)**(channel_load - 1)
    den = (ac_num_channel_uses * ac_range_num_slots) + (tr_duration + ack_duration)

    return num/den

########################################
# General parameters
########################################

# Define channel load
channel_load = 5

########################################
# Training parameters
########################################

# Number of training channel uses
tr_num_channel_uses = 1000

# Number of training slots
tr_num_slots = 150

# Compute training duration
tr_duration = tr_num_channel_uses * tr_num_slots

########################################
# ACK parameters
########################################

# Number of ACK channel uses
ack_num_channel_uses = 1000

# Number of ACK slots
ack_num_slots = channel_load

# Compute ACK duration
ack_duration = ack_num_channel_uses * ack_num_slots

########################################
# Simulation
########################################

# Number of access channel uses
ac_num_channel_uses = 1000

# Range of the number of access slots
ac_range_num_slots = np.arange(1, 1000)

# Compute expected throughput
expected_throughput = compute_expected_throughput(channel_load, ac_range_num_slots, ac_num_channel_uses, tr_duration, ack_duration)

fig, ax = plt.subplots()

ax.plot(ac_range_num_slots, expected_throughput)

plt.show()
