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
data = np.load('../data/figure5b.npz')

range_rec_error = data['range_rec_error']
channel_loads = data['channel_loads']
access_policies = data['access_policies']

proba_access = data['proba_access']

# Compute average
avg_proba_access = np.nanmean(proba_access, axis=-1)

# Smoothed channel load
sm_channel_loads = np.linspace(channel_loads.min(), channel_loads.max(), 501)

# Smooth data
sm_avg_proba_access = np.zeros((avg_proba_access.shape[0], avg_proba_access.shape[1], 501))

# Go through all access policies
for ap, access_policy in enumerate(access_policies):

    # Go through all reconstruction errors
    for re, rec_error in enumerate(range_rec_error):

            model = scipy.interpolate.interp1d(channel_loads, avg_proba_access[ap, re, :], kind = "cubic")
            sm_avg_proba_access[ap, re] = model(sm_channel_loads)

##################################################
# Plot
##################################################

# Define error labels
error_labels = [
    'true',
    '$\overline{\mathrm{SE}}=10^{-1}$',
    #'$\overline{\mathrm{SE}}=10^{-2}$',
    '$\overline{\mathrm{SE}}=10^{-3}$'
    ]

# Define error markers
markers = ['o', 's', None, '^']
linestyles = ['--', '-.', ':']

#colors = ['#7f58af', '#64c5eb', '#e84d8a', '#feb326']
#colors = ['#674a40', '#50a3a4', '#fcaf38', '#f95335']

fig, ax = plt.subplots(figsize=(6.5/2, 3))

# Go through all access policies
for ap, access_policy in enumerate(access_policies):

    plt.gca().set_prop_cycle(None)

    # Go through all reconstruction errors
    for re, rec_error in enumerate(range_rec_error):

        if re == 0:
            ax.plot(sm_channel_loads, sm_avg_proba_access[ap, re, :], linewidth=1.5, linestyle=linestyles[ap], marker=markers[re], markevery=125, markersize=5, markerfacecolor='white', color='black')
            continue

        if re == 2:
            continue

        ax.plot(sm_channel_loads, sm_avg_proba_access[ap, re, :], linewidth=1.5, linestyle=linestyles[ap], marker=markers[re], markevery=125, markersize=5, markerfacecolor='white')

# Legend plots
ap1, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 0, :], linewidth=1.5, linestyle=linestyles[0], color='grey', label=access_policies[0])
ap2, = ax.plot(sm_channel_loads, sm_avg_proba_access[1, 0, :], linewidth=1.5, linestyle=linestyles[1], color='grey', label=access_policies[1])
ap3, = ax.plot(sm_channel_loads, sm_avg_proba_access[2, 0, :], linewidth=1.5, linestyle=linestyles[2], color='grey', label=access_policies[2])

r1, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 0, :], linewidth=0.0, linestyle=None, marker=markers[0], markevery=125, markersize=5, markerfacecolor='white', color='black', label='true')
plt.gca().set_prop_cycle(None)
r2, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 0, :], linewidth=0.0, linestyle=None, marker=markers[1], markevery=125, markersize=5, markerfacecolor='white', label=r'$\overline{\mathrm{SE}}=10^{-1}$')
r3, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 0, :], linewidth=0.0, linestyle=None, marker=markers[3], markevery=125, markersize=5, markerfacecolor='white', label=r'$\overline{\mathrm{SE}}=10^{-3}$')

ax.set_xlabel('channel load, $\kappa$')
ax.set_ylabel('probability of access')

# ax.set_xscale('log')
# ax.set_yscale('log')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

plt.subplots_adjust(
	left = 0.15,  	# the left side of the subplots of the figure
	right = 0.99,   # the right side of the subplots of the figure
	bottom = 0.26,   # the bottom of the subplots of the figure
	top = 0.99,     # the top of the subplots of the figure
	wspace = 0.5,  	# the amount of width reserved for space between subplots,
    	           	# expressed as a fraction of the average axis width
	hspace = 0.05   # the amount of height reserved for space between subplots,
              	 	# expressed as a fraction of the average axis height
              )

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + (box.height * 0.125),
#                  box.width, box.height * 0.9])

ax.legend(fontsize='small', ncol=3, framealpha=0.5,
        bbox_to_anchor=[0.95, -0.15], fancybox=True)

ax.lines.remove(ap1)
ax.lines.remove(ap2)
ax.lines.remove(ap3)

ax.lines.remove(r1)
ax.lines.remove(r2)
ax.lines.remove(r3)

#plt.tight_layout()

plt.savefig('../figs/figure5b.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure5b.tex")

plt.show()
