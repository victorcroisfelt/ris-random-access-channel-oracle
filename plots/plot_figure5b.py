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
data = np.load('../data/figure5b.npz')

num_successful_attempts = data['num_successful_attempts']
channel_loads = data['channel_loads']
range_rec_error = data['range_rec_error']
access_policies = data['access_policies']

##################################################
# Plot
##################################################

# Define error labels
error_labels = [
    'true',
    '$\overline{\mathrm{SE}}=10^{-1}$',
    '$\overline{\mathrm{SE}}=10^{-3}$'
    ]

# Define error markers
error_markers = [None, 'o', 's', 'd']

# fig, ax = plt.subplots()
#
# ax.plot(channel_loads, np.nanmean(num_successful_attempts[0, 0, :, :], axis=-1), linewidth=1.5, linestyle='-', label='RCURAP', color='black')
#
# # Go through all recontruction errors
# for re in range(range_rec_error.size):
#     ax.plot(channel_loads, np.nanmean(num_successful_attempts[-1, re, :, :], axis=-1), marker=error_markers[re], linewidth=1.5, linestyle='--', label='SMAP: ' + error_labels[re])
#
# ax.set_xlabel(r'channel load, $\kappa$')
# ax.set_ylabel(r'successful attempts ratio')
#
# #ax.set_yscale('log')
#
# ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
#
# ax.legend(fontsize='medium', loc='best')
#
# plt.tight_layout()
#
# plt.savefig('../figs/figure5b.pdf', dpi='figure', format='pdf', transparent='True')
#
# tikzplotlib.save("../tikz/figure5b.tex")
#
# plt.show()

# fig, ax = plt.subplots()
#
# data = num_successful_attempts[-1].reshape((14, -1))
#
# for re in range(range_rec_error.size):
#
#     filtered_data = data[re, ~np.isnan(data[re])]
#     ax.boxplot(filtered_data, positions=[re])
#
# plt.show()

fig, ax = plt.subplots()

# Go through all channel loads
for cl in [1, 2, 3]:
    #ax.plot(range_rec_error, np.ones_like(range_rec_error)*np.nanmean(num_successful_attempts[0, 0, cl, :], axis=-1), linewidth=1.5, linestyle='-', label='RCURAP', color='black')
    ax.plot(range_rec_error, np.nanmean(num_successful_attempts[cl, :, :, :], axis=(-1, -2)), linewidth=1.5, linestyle='--') #label='SMAP: ' + error_labels[re])

#marker=error_markers[re]
ax.set_xlabel(r'channel load, $\kappa$')
ax.set_ylabel(r'successful attempts ratio')

ax.set_xscale('log')
ax.set_yscale('log')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

ax.legend(fontsize='medium', loc='best')

plt.tight_layout()

plt.savefig('../figs/figure5b.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure5b.tex")

#plt.show()
