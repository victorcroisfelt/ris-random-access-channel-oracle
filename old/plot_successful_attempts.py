import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

########################################
# Loading
########################################

# Load data
aloha_blocked_data = np.load('data/aloha_blocked_contending.npz')
strongest_data = np.load('data/ris_strongest_N100_contending.npz')
rand_data = np.load('data/ris_rand_N100_contending.npz')

# Number of successful access attempts
total_num_successful_attempts_aloha_blocked = aloha_blocked_data['total_num_successful_attempts']
total_num_successful_attempts_strongest = strongest_data['total_num_successful_attempts']
total_num_successful_attempts_rand = rand_data['total_num_successful_attempts']

# Common parameters
num_configs_range = strongest_data['num_configs_range']
num_ues_range = strongest_data['num_ues_range']

# Stack
num_successful_attempts = np.stack((total_num_successful_attempts_strongest, total_num_successful_attempts_rand,
                                    total_num_successful_attempts_aloha_blocked), axis=0)

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 14})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots(figsize=(3.15, 2.3))

markers = ['o', 's', '*']
markersizes = [4, 6, 8]

colors = ['#7a5195', '#ef5675', 'black']
styles = ['-', '--', ':']
methods = ['SCP', 'CARP', 'URP']

marker_start = [0, 1, 2]

# markers = ['o', 's', 'd']
# colors = ['black', '#7a5195', '#ef5675', '#ffa600']
# styles = ['-', '--', '-.', ':']
# methods = ['slotted ALOHA', 'slotted ALOHA: blocked', 'A) strongest policy', 'B) random policy']

# Go through all methods
for mm in range(3):

    # Legend use
    ax.plot(num_ues_range, np.nanmean(num_successful_attempts[mm, 0], axis=-1)/num_configs_range[0], linewidth=2, linestyle=styles[mm],
            color=colors[mm])

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        if num_configs == 4:
            continue

        ax.plot(num_ues_range, np.nanmean(num_successful_attempts[mm, cc], axis=-1)/num_configs_range[cc], linewidth=2,
                marker=markers[cc], markevery=2, markersize=markersizes[cc], linestyle=styles[mm], color=colors[mm])

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    if num_configs == 4:
        continue

    # Legend use
    ax.plot(num_ues_range, np.nanmean(num_successful_attempts[0, cc], axis=-1)/num_configs_range[cc], linewidth=None,
            linestyle=None, markevery=1, marker=markers[cc], markersize=markersizes[cc], color='black',
            label='$S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel(r'number of contending UEs, $K$')
ax.set_ylabel(r'avg. $\mathrm{SA}$ [pkt./slot]')

# Legend
ax.legend(fontsize='x-small', framealpha=0.5, loc=1)

ax.set_xticks(np.arange(1, 11))

# Pop out some useless legend curves
ax.lines.pop(-1)
ax.lines.pop(-1)

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

#plt.savefig('figs/num_successful_attempts.pdf', dpi=300)

plt.show()