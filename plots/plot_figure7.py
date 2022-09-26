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
data = np.load('../data/figure7.npz')

channel_loads = data['channel_loads']
access_policies = data['access_policies']
ack_methods = data['ack_methods']

proba_access = data['proba_access']
throughput = data['throughput']

# Compute averages
avg_proba_access = np.nanmean(proba_access, axis=-1)
avg_throughput = np.nanmean(throughput, axis=-1)

# Smoothed channel load
sm_channel_loads = np.linspace(channel_loads.min(), channel_loads.max(), 150)

# Smooth data
sm_avg_proba_access = np.zeros((avg_proba_access.shape[0], avg_proba_access.shape[1], 150))
sm_avg_throughput = np.zeros((avg_throughput.shape[0], avg_proba_access.shape[1], 150))

# Go through all access policies
for ap, access_policy in enumerate(access_policies):

    # Go through all ack methods
    for ack in range(3):

        model = scipy.interpolate.interp1d(channel_loads, avg_proba_access[ack, ap, :], kind = "cubic")

        sm_avg_proba_access[ack, ap, :] = model(sm_channel_loads)

        model = scipy.interpolate.interp1d(channel_loads, avg_throughput[ack, ap, :], kind = "cubic")

        sm_avg_throughput[ack, ap, :] = model(sm_channel_loads)

##################################################
# Plot
##################################################

# Define error markers
linestyles = ['-', '--', '-.', ':']
markers = ['d', 'd', 'v']

colors = ['black', 'tab:blue', 'tab:orange', 'tab:green']
#colors = ['#674a40', '#50a3a4', '#fcaf38', '#f95335']

masks = [[0, 49, 99, 149], [9, 59, 109], [19, 69, 119]]
#
# fig, ax = plt.subplots(figsize=(6.5/2, 3))
#
# # Go through all access policies
# for ap, access_policy in enumerate(access_policies):
#
#     # Go through all ack methods
#     for ack in range(3):
#
#         ax.plot(sm_channel_loads, sm_avg_proba_access[ack, ap, :], linewidth=1.5, linestyle=linestyles[ap], color=colors[ap])
#         ax.plot(sm_channel_loads[masks[ack]], sm_avg_proba_access[ack, ap, masks[ack]], linewidth=0.0, marker=markers[ack], markerfacecolor='white', markersize=5, color=colors[ap])
#
# # Legend plots
# ap1, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 0, :], linewidth=1.5, linestyle=linestyles[0], color=colors[0], label=access_policies[0])
# ap2, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 1, :], linewidth=1.5, linestyle=linestyles[1], color=colors[1], label=access_policies[1])
# ap3, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 2, :], linewidth=1.5, linestyle=linestyles[2], color=colors[2], label=access_policies[2])
# ap4, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 2, :], linewidth=1.5, linestyle=linestyles[3], color=colors[3], label=access_policies[3])
#
# r1, = ax.plot(sm_channel_loads, sm_avg_proba_access[0, 0, :], linewidth=0.0, linestyle=None, marker=markers[0], markerfacecolor='white', markevery=125, markersize=5, color='black', label='perfect ACK')
# r2, = ax.plot(sm_channel_loads, sm_avg_proba_access[1, 0, :], linewidth=0.0, linestyle=None, marker=markers[1], markerfacecolor='white', markevery=125, markersize=5, color='black', label='precoding-based')
# r3, = ax.plot(sm_channel_loads, sm_avg_proba_access[2, 0, :], linewidth=0.0, linestyle=None, marker=markers[2], markerfacecolor='white', markevery=125, markersize=5, color='black', label=r'TDMA-based')
#
# ax.set_xlabel('channel load, $\kappa$')
# ax.set_ylabel('probability of access')
#
# ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
#
# plt.tight_layout()
#
# box = ax.get_position()
# ax.set_position([box.x0, box.y0 + (box.height * 0.125),
#                  box.width, box.height * 0.9])
#
# ax.legend(fontsize='small', ncol=3, framealpha=0.5,
#         bbox_to_anchor=[1.0, -0.175], fancybox=True)
#
# ax.lines.remove(ap1)
# ax.lines.remove(ap2)
# ax.lines.remove(ap3)
# ax.lines.remove(ap4)
#
# ax.lines.remove(r1)
# ax.lines.remove(r2)
# ax.lines.remove(r3)
#
# plt.savefig('../figs/figure7_proba_access.pdf', dpi='figure', format='pdf', transparent='True')
#
# tikzplotlib.save("../tikz/figure7_proba_access.tex")
#
# plt.show()


######


fig, ax = plt.subplots(figsize=(6.5/2, 3))

# Go through all access policies
for ap, access_policy in enumerate(access_policies):

    # Go through all ack methods
    for ack in [1, 2]:

        ax.plot(sm_channel_loads, sm_avg_throughput[ack, ap, :], linewidth=1.5, linestyle=linestyles[ap], color=colors[ap])
        ax.plot(sm_channel_loads[masks[ack]], sm_avg_throughput[ack, ap, masks[ack]], linewidth=0.0, marker=markers[ack], markerfacecolor='white', markersize=5, color=colors[ap])

# Legend plots
ap1, = ax.plot(sm_channel_loads, sm_avg_throughput[1, 0, :], linewidth=1.5, linestyle=linestyles[0], color=colors[0], label=access_policies[0])
ap2, = ax.plot(sm_channel_loads, sm_avg_throughput[1, 1, :], linewidth=1.5, linestyle=linestyles[1], color=colors[1], label=access_policies[1])
ap3, = ax.plot(sm_channel_loads, sm_avg_throughput[1, 2, :], linewidth=1.5, linestyle=linestyles[2], color=colors[2], label=access_policies[2])
ap4, = ax.plot(sm_channel_loads, sm_avg_throughput[1, 3, :], linewidth=1.5, linestyle=linestyles[3], color=colors[3], label=access_policies[3])

#r1, = ax.plot(sm_channel_loads, sm_avg_throughput[0, 0, :], linewidth=0.0, linestyle=None, marker=markers[0], markerfacecolor='white', markevery=125, markersize=5, color='grey', label='perfect ACK')
r2, = ax.plot(sm_channel_loads, sm_avg_throughput[1, 0, :], linewidth=0.0, linestyle=None, marker=markers[1], markerfacecolor='white', markevery=125, markersize=5, color='grey', label='prec. ACK')
r3, = ax.plot(sm_channel_loads, sm_avg_throughput[2, 0, :], linewidth=0.0, linestyle=None, marker=markers[2], markerfacecolor='white', markevery=125, markersize=5, color='grey', label=r'TDMA ACK')

ax.set_xlabel('channel load, $\kappa$')
ax.set_ylabel('throughput')

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

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
# ax.set_position([box.x0, box.y0 + (box.height * 0.085),
#                  box.width, box.height * 0.970])

ax.legend(fontsize='small', ncol=3, framealpha=0.5,
        bbox_to_anchor=[0.95, -0.15], fancybox=True, columnspacing=0.75)

ax.lines.remove(ap1)
ax.lines.remove(ap2)
ax.lines.remove(ap3)
ax.lines.remove(ap4)

#ax.lines.remove(r1)
ax.lines.remove(r2)
ax.lines.remove(r3)

ax.tick_params(axis='both', which='major', labelsize=8)
ax.tick_params(axis='both', which='minor', labelsize=8)

plt.savefig('../figs/figure7_throughput.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure7_throughput.tex")

plt.show()
