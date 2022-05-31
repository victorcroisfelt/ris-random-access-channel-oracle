import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

########################################
# Loading
########################################

# Load data
strongest_data = np.load('data/ris_strongest_N100_slot_based.npz')
rand_data = np.load('data/ris_rand_N100_slot_based.npz')
aloha_data = np.load('data/ris_aloha_N100_slot_based.npz')

# Number of successful access attempts
total_num_successful_attempts_strongest = strongest_data['avg_num_successful_attempts']
total_num_successful_attempts_rand = rand_data['avg_num_successful_attempts']
total_num_successful_attempts_aloha = aloha_data['avg_num_successful_attempts']

# Common parameters
num_configs_range = strongest_data['num_configs_range']
channel_load_range = strongest_data['channel_load_range']

# Stack
avg_num_successful_attempts = np.stack(
    (
        np.nanmean(total_num_successful_attempts_strongest, axis=-1),
        np.nanmean(total_num_successful_attempts_rand, axis=-1),
        np.nanmean(total_num_successful_attempts_aloha, axis=-1)
    ),
    axis=0)

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 14})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots(figsize=(2*3.15, 1.5*2.3))

markers = ['o', '*']
markersizes = [4, 8]
marker_start = [0, 1]

colors = ['#7a5195', '#ef5675', 'black']
styles = ['-', '--', ':']
methods = ['SCP', 'CARP', 'URP']

config_index = [num_configs_range.tolist().index(2), num_configs_range.tolist().index(8)]

# Go through all access policies
for mm in range(3):

    # Legend use
    ax.plot(channel_load_range, avg_num_successful_attempts[mm, config_index[0], :] / num_configs_range[config_index[0]],
            linewidth=2, linestyle=styles[mm], color=colors[mm])

    # Go through all number of configurations
    for idx, index in enumerate(config_index):

        ax.plot(channel_load_range, avg_num_successful_attempts[mm, index, :] / num_configs_range[index],
                linewidth=2, linestyle=styles[mm], color=colors[mm],
                marker=markers[idx], markevery=2, markersize=markersizes[idx])

# Go through all number of configurations
for idx, index in enumerate(config_index):

    # Legend use
    ax.plot(channel_load_range, avg_num_successful_attempts[0, index, :] / num_configs_range[index], linewidth=None,
            linestyle=None, markevery=1, marker=markers[idx], markersize=markersizes[idx], color='black',
            label='$S =' + str(num_configs_range[index]) + '$')

# Set axis
ax.set_xlabel(r'channel load, $\kappa$')
ax.set_ylabel(r'avg. $\mathrm{SA}$ [pkt./slot]')

# Legend
ax.legend(fontsize='x-small', framealpha=0.5, loc=1)

ax.set_xticks(np.arange(1, 11))

# # Pop out some useless legend curves
ax.lines.pop(-1)
ax.lines.pop(-1)

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

#plt.savefig('figs/num_successful_attempts.pdf', dpi=300)

plt.show()