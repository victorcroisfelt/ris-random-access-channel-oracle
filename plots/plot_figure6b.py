import scipy
import numpy as np

import matplotlib
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
data = np.load('../data/figure6.npz')

channel_loads = data['channel_loads']
access_policies = data['access_policies']

proba_access = data['proba_access']
throughput = data['throughput']

# Compute averages
avg_proba_access = np.nanmean(proba_access, axis=-1)

# Create tsw vector
tsw_vec = np.linspace(0, 1)

# Create denominator
den = 2 + 2 *tsw_vec


########################################
# Plot
########################################
breakpoint()
# Define AP identifiers
linestyles = ['-', '--', '-.', ':']

colors = ['black', 'tab:blue', 'tab:orange', 'tab:green']
labels = ['baseline', 'proposed: $R$-CARAP', 'proposed: $R$-GSCAP', 'proposed: SMAP']

fig, ax = plt.subplots(figsize=(6.1, 3), sharey=True)

# Go through all access policies
for ap, access_policy in enumerate(access_policies):

    if ap == 0:

        ax.plot(tsw_vec, (avg_proba_access[ap, 1] / 2) * np.ones_like(tsw_vec), linewidth=1.5, linestyle=linestyles[ap], color=colors[ap],
                label=labels[ap])

    else:

        ax.plot(tsw_vec, avg_proba_access[ap, 1] / den, linewidth=1.5, linestyle=linestyles[ap], color=colors[ap],
                label=labels[ap])

ax.set_xlabel(r'switching-symbol time ratio, $\dfrac{T_{\rm sw}}{T_{\rm symb}}$')
ax.set_ylabel(r'expected overall throughput, $\overline{\mathrm{TP}}$')

ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

#ax.set_yscale('log')

ax.legend(fontsize='small', framealpha=0.5, fancybox=True)

plt.tight_layout()

# plt.subplots_adjust(
#     left=0.09,
#     right=0.99,
#     bottom=0.125,
#     top=0.925,
#     wspace=0.05,
#     hspace=0.05
# )

plt.savefig('../figs/figure6b.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure6b.tex")

plt.show()

