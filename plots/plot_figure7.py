import scipy
import numpy as np

import matplotlib.pyplot as plt

from matplotlib_config import plot_rc_config

import tikzplotlib

########################################
# Preamble
########################################
plot_rc_config()

########################################
# Load data
########################################
data = np.load('../data/figure7.npz')

channel_loads = data['channel_loads']
access_policies = data['access_policies']
ack_methods = data['ack_methods']

proba_access = data['proba_access']

# Compute averages
avg_proba_access = np.squeeze(np.nanmean(proba_access, axis=-1))
avg_proba_access = avg_proba_access[1:]

# Smoothed channel load
sm_channel_loads = np.linspace(1, 10, 150)

# Smooth data
sm_avg_proba_access = np.zeros((avg_proba_access.shape[0], 150))

# Go through all access policies
for ap in range(3):
    model = scipy.interpolate.interp1d(channel_loads, avg_proba_access[ap, :], kind="cubic")
    sm_avg_proba_access[ap, :] = model(sm_channel_loads)

########################################
# Plot
########################################

# Define AP identifiers
linestyles = ['-', '--', '-.', ':']

colors = ['black', 'tab:blue', 'tab:orange']
labels = ['random configs.', 'precoding-based', 'scheduled-based']

fig, ax = plt.subplots(figsize=(6.1, 3), sharey=True)

# Go through all access policies
for ap in range(3):
    ax.plot(sm_channel_loads, sm_avg_proba_access[ap, :], linewidth=1.5, linestyle=linestyles[ap], color=colors[ap],
            label=labels[ap])

ax.set_xlabel(r'channel load, $\kappa$')
ax.set_ylabel(r'expected overall throughput, $\overline{\mathrm{TP}}$')

ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

ax.legend(fontsize='small', framealpha=0.5, fancybox=True)

plt.tight_layout()

plt.savefig('../figs/figure7.pdf', dpi='figure', format='pdf', transparent='True')
tikzplotlib.save("../tikz/figure7.tex")

plt.show()

