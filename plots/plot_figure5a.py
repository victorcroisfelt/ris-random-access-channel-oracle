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
data = np.load('../data/figure5a.npz')

reconstruction_nmse = data['reconstruction_nmse']
range_tr_num_slots = data['range_tr_num_slots']
range_mvu_error_dl = data['range_mvu_error_dl']

##################################################
# Plot
##################################################

# Define bounds according to Nyquist-Shannon theorem
bounds = [16, 46, 142, 150]

# Define bound labels
bound_labels = [
    r'$\acute{N}^{10^{-2}}_{\rm tr}=16$',
    r'$\acute{N}^{10^{-3}}_{\rm tr}=46$',
    r'$\check{N}^{10^{-2}}_{\rm tr}=142$',
    r'$\tilde{N}_{\rm tr}=150$'
]

# Define error labels
error_labels = [
    'true',
    '$\delta_{\mathrm{tol}}^{\scriptscriptstyle \mathrm{DL}}=10^{-1}$',
    '$\delta_{\mathrm{tol}}^{\scriptscriptstyle \mathrm{DL}}=10^{-2}$',
    '$\delta_{\mathrm{tol}}^{\scriptscriptstyle \mathrm{DL}}=10^{-3}$'
    ]

# Define error markers
error_markers = [None, 'o', 's', 'd']

fig, ax = plt.subplots(figsize=(6.5/2, 3))

# Go through all error tolerances
for et in range(range_mvu_error_dl.size):

    if et == 0:
        ax.plot(range_tr_num_slots, np.nanmean(reconstruction_nmse[et, :, :], axis=-1), linewidth=1.5, label=error_labels[et], color='black')
    else:
        ax.plot(range_tr_num_slots, np.nanmean(reconstruction_nmse[et, :, :], axis=-1), linestyle='--', markersize=4, marker=error_markers[et], markevery=10, linewidth=1.5, label=error_labels[et])

for bb, bound in enumerate(bounds):
    ax.plot([bound, bound], [1e-4, 1], linestyle=':', color='black', alpha=0.5)

ax.text(20, 0.1, bound_labels[0], fontsize='small', rotation='vertical')
ax.text(50, 0.1, bound_labels[1], fontsize='small', rotation='vertical')
ax.text(85, 0.005, bound_labels[2], fontsize='small')
ax.text(154, 0.0003, bound_labels[3], fontsize='small', rotation='vertical')

ax.set_xlabel(r'number of training slots, $N_{\rm tr}$')
ax.set_ylabel(r'normalized $\overline{\mathrm{SE}}$ in (33)')

ax.set_ylim([1e-4, 1])

ax.set_yscale('log')

ax.legend(fontsize='medium', loc='best', framealpha=0.5)

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.savefig('../figs/figure5a.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure5a.tex")

plt.show()
