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
data = np.load('../data/figure8.npz')

channel_loads = data['channel_loads']
access_policies = data['access_policies']
ack_methods = data['ack_methods']

switch_times = data['switch_times']

proba_access = data['proba_access']
throughput = data['throughput']

# Compute averages
avg_proba_access = np.nanmean(proba_access, axis=-1)
avg_throughput = np.nanmean(throughput, axis=-1)

# Smoothed channel load
sm_channel_loads = np.linspace(channel_loads.min(), channel_loads.max(), 150)

# Smooth data
sm_avg_throughput = np.zeros((avg_throughput.shape[0], avg_proba_access.shape[1], 150))

# Go through all switch times
for st, switch_time in enumerate(switch_times):

    # Go through all ack methods
    for ack in range(3):

        model = scipy.interpolate.interp1d(channel_loads, avg_throughput[st, ack, 2, :], kind = "cubic")

        sm_avg_throughput[st, ack, :] = model(sm_channel_loads)

##################################################
# Plot
##################################################

# Define error markers
markers = ['None', 'd', 'v']

#colors = ['black', 'tab:blue', 'tab:orange', 'tab:green']
colors = ['#674a40', '#50a3a4', '#fcaf38', '#f95335']

masks = [[0, 49, 99, 149], [9, 59, 109], [19, 69, 119]]

fig, ax = plt.subplots(figsize=(6.5/2, 3))

# Go through all switch times
for st, switch_time in enumerate(switch_times):

    # Go through all ack methods
    for ack in range(3):

        ax.plot(sm_channel_loads, sm_avg_throughput[st, ack, :], linewidth=1.5, linestyle='-.', color=colors[st])


# Go through all switch times
for st, switch_time in enumerate(switch_times):

    # Go through all ack methods
    for ack in range(3):

        ax.plot(sm_channel_loads[masks[ack]], sm_avg_throughput[st, ack, masks[ack]], linewidth=0.0, marker=markers[ack], markerfacecolor='white', markersize=5, color=colors[st])



# Legend plots
l1, = ax.plot(sm_channel_loads, sm_avg_throughput[0, 0, :], linewidth=1.5, linestyle='-.', color='grey', label='RGSCAP w/ perfect ACK')
l2, = ax.plot(sm_channel_loads, sm_avg_throughput[0, 1, :], linewidth=1.5, linestyle='-.', marker=markers[1], markerfacecolor='white', markersize=4, color='grey', label='RGSCAP w/ prec. ACK')
l3, = ax.plot(sm_channel_loads, sm_avg_throughput[0, 2, :], linewidth=1.5, linestyle='-.', marker=markers[2], markerfacecolor='white', markersize=4, color='grey', label='RGSCAP w/ TDMA ACK')

# #r1, = ax.plot(sm_channel_loads, sm_avg_throughput[0, 0, :], linewidth=0.0, linestyle=None, marker=markers[0], markerfacecolor='white', markevery=125, markersize=4, color='grey', label='perfect ACK')
# r2, = ax.plot(sm_channel_loads, sm_avg_throughput[1, 0, :], linewidth=0.0, linestyle=None, marker=markers[1], markerfacecolor='white', markevery=125, markersize=4, color='grey', label='prec. ACK')
# r3, = ax.plot(sm_channel_loads, sm_avg_throughput[2, 0, :], linewidth=0.0, linestyle=None, marker=markers[2], markerfacecolor='white', markevery=125, markersize=4, color='grey', label=r'TDMA ACK')

ax.set_xlabel('channel load, $\kappa$')
ax.set_ylabel('throughput')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.subplots_adjust(
	left = 0.175,  	# the left side of the subplots of the figure
	right = 0.99,   # the right side of the subplots of the figure
	bottom = 0.125,  # the bottom of the subplots of the figure
	top = 0.99,     # the top of the subplots of the figure
	wspace = 0.5,  	# the amount of width reserved for space between subplots,
    	           	# expressed as a fraction of the average axis width
	hspace = 0.05   # the amount of height reserved for space between subplots,
              	 	# expressed as a fraction of the average axis height
              )

# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + (box.height * 0.085),
#                  box.width, box.height * 0.970])


ax.legend(fontsize='x-small', framealpha=0.5, fancybox=True)

ax.lines.remove(l1)
ax.lines.remove(l2)
ax.lines.remove(l3)

ax.plot([40, 40], [0.075, 0.120], linestyle='--', color='black', alpha=.3)
ax.text(10, 0.11, r'$T_{\rm config}=0$', fontsize=8)

ax.plot([60, 60], [0.035, 0.065], linestyle='--', color='black', alpha=.3)
ax.text(62.5, 0.04, r'$T_{\rm config}=1$', fontsize=8)

ax.plot([100, 100], [0.01, 0.030], linestyle='--', color='black', alpha=.3)
ax.text(102.5, 0.0125, r'$T_{\rm config}=1$', fontsize=8)

# ax.lines.remove(ap4)
#
# #ax.lines.remove(r1)
# ax.lines.remove(r2)
# ax.lines.remove(r3)

ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

plt.savefig('../figs/figure7_throughput.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure7_throughput.tex")

plt.show()
