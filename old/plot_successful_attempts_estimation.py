import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

########################################
# Loading
########################################

# Load data
strongest_data = np.load('data/ris_strongest_N100_estimation.npz')
rand_data = np.load('data/ris_rand_N100_estimation.npz')

# Number of successful access attempts
total_num_successful_attempts_strongest = strongest_data['total_num_successful_attempts']
total_num_successful_attempts_rand = rand_data['total_num_successful_attempts']

# Common parameters
estimation_quality_range = strongest_data['estimation_quality_range']
num_configs_range = strongest_data['num_configs_range']
num_ues_range = strongest_data['num_ues_range']

# Stack
num_successful_attempts = np.stack((total_num_successful_attempts_strongest, total_num_successful_attempts_rand), axis=0)

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 14})
rc('text', usetex=True)

# Open axes
#fig, ax = plt.subplots(figsize=(3.15, 2.3))

fig, ax = plt.subplots()

markers = ['o', '*']
markersizes = [4, 8]

colors = ['#7a5195', '#ef5675']
styles = ['-', '--']
methods = ['SCP', 'CARP']

marker_start = [0, 1]

# Go through all methods
for mm in range(2):

    # Legend use
    ax.plot(estimation_quality_range, np.nanmean(num_successful_attempts[mm, :, 0], axis=(-1, -2))/num_configs_range[0], linewidth=2, linestyle=styles[mm],
            color=colors[mm])

    # Go through all number of configurations
    for cc, num_configs in enumerate(num_configs_range):

        if num_configs == 4:
            continue

        ax.plot(estimation_quality_range, np.nanmean(num_successful_attempts[mm, :, cc], axis=(-1, -2))/num_configs_range[cc], linewidth=2,
                marker=markers[cc], markevery=2, markersize=markersizes[cc], linestyle=styles[mm], color=colors[mm])

# Go through all number of configurations
for cc, num_configs in enumerate(num_configs_range):

    # Legend use
    ax.plot(estimation_quality_range, np.nanmean(num_successful_attempts[0, :, cc], axis=(-1, -2))/num_configs_range[cc], linewidth=None,
            linestyle=None, markevery=1, marker=markers[cc], markersize=markersizes[cc], color='black',
            label='$S =' + str(num_configs) + '$')

# Set axis
ax.set_xlabel(r'estimation quality range, $\epsilon$')
ax.set_ylabel(r'avg. $\mathrm{SA}$ [pkt./slot]')

# Legend
ax.legend(fontsize='x-small', framealpha=0.5, loc=1)

#ax.set_xticks(np.arange(1, 11))

# Pop out some useless legend curves
ax.lines.pop(-1)
ax.lines.pop(-1)

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

#plt.savefig('figs/num_successful_attempts.pdf', dpi=300)

plt.show()