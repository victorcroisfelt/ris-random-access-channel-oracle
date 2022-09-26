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
avg_throughput = np.nanmean(throughput, axis=-1)

# Smoothed channel load
sm_channel_loads = np.linspace(channel_loads.min(), channel_loads.max(), 501)

# Smooth data
sm_avg_proba_access = np.zeros((avg_proba_access.shape[0], 501))
sm_avg_throughput = np.zeros((avg_throughput.shape[0], 501))

# Go through all access policies
for ap, access_policy in enumerate(access_policies):

        model = scipy.interpolate.interp1d(channel_loads, avg_proba_access[ap, :], kind = "cubic")

        sm_avg_proba_access[ap, :] = model(sm_channel_loads)

        model = scipy.interpolate.interp1d(channel_loads, avg_throughput[ap, :], kind = "cubic")

        sm_avg_throughput[ap, :] = model(sm_channel_loads)

##################################################
# Plot
##################################################

# Define AP identifiers
linestyles = ['-', '--', '-.', ':']
colors = ['black', 'tab:blue', 'tab:orange', 'tab:green']

#colors = ['#7f58af', '#64c5eb', '#e84d8a', '#feb326']
#colors = ['#674a40', '#50a3a4', '#fcaf38', '#f95335']

# fig, ax = plt.subplots(figsize=(6.5/2, 3))
#
# # Go through all access policies
# for ap, access_policy in enumerate(access_policies):
#
#     if ap == 0:
#         ax.plot(sm_channel_loads, sm_avg_proba_access[ap, :], linewidth=1.5, linestyle=linestyles[ap], color='black', label=access_policy)
#         continue
#
#     ax.plot(sm_channel_loads, sm_avg_proba_access[ap, :], linewidth=1.5, linestyle=linestyles[ap], label=access_policy)
#
# ax.set_xlabel('channel load, $\kappa$')
# ax.set_ylabel('probability of access')
#
# ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
#
# plt.tight_layout()
#
# ax.legend(fontsize='small', framealpha=0.5, fancybox=True)
#
# plt.tight_layout()
#
# plt.savefig('../figs/figure6_proba_access.pdf', dpi='figure', format='pdf', transparent='True')
#
# tikzplotlib.save("../tikz/figure6_proba_access.tex")
#
# plt.show()


#####


# fig, ax = plt.subplots(figsize=(6.1, 3), ncols=3)
#
# # Go through all access policies
# for ap, access_policy in enumerate(access_policies):
#
#     if ap == 0:
#         ax.plot(sm_channel_loads, sm_avg_throughput[ap, :], linewidth=1.5, linestyle=linestyles[ap], color='black', label=access_policy)
#         continue
#
#     ax.plot(sm_channel_loads, sm_avg_throughput[ap, :], linewidth=1.5, linestyle=linestyles[ap], label=access_policy)
#
# ax.set_xlabel('channel load, $\kappa$')
# ax.set_ylabel('throughput')
#
# ax.tick_params(axis='both', which='major', labelsize=8)
# ax.tick_params(axis='both', which='minor', labelsize=8)
#
# ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
#
# ax.legend(fontsize='small', framealpha=0.5, fancybox=True)
#
# plt.subplots_adjust(
# 	left = 0.15,  	# the left side of the subplots of the figure
# 	right = 0.99,   # the right side of the subplots of the figure
# 	bottom = 0.125,   # the bottom of the subplots of the figure
# 	top = 0.99,     # the top of the subplots of the figure
# 	wspace = 0.5,  	# the amount of width reserved for space between subplots,
#     	           	# expressed as a fraction of the average axis width
# 	hspace = 0.05   # the amount of height reserved for space between subplots,
#               	 	# expressed as a fraction of the average axis height
#               )
#
# plt.savefig('../figs/figure6_throughput.pdf', dpi='figure', format='pdf', transparent='True')
#
# tikzplotlib.save("../tikz/figure6_throughput.tex")
#
# plt.show()

fig, axes = plt.subplots(figsize=(6.1, 3), ncols=3, sharey=True)

# Go through all access policies
for ap, access_policy in enumerate(access_policies):
    axes[0].plot(sm_channel_loads, sm_avg_throughput[ap, :], linewidth=1.5, linestyle=linestyles[ap], color=colors[ap], label=access_policy)

axes[0].set_xlabel('channel load, $\kappa$')
axes[0].set_ylabel('throughput')

axes[0].tick_params(axis='both', which='major', labelsize=8)
axes[0].tick_params(axis='both', which='minor', labelsize=8)

axes[0].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

axes[0].legend(fontsize='small', framealpha=0.5, fancybox=True)

axes[0].set_title('perfect ACK', fontsize=10)

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

# Go through all access policies
for ap, access_policy in enumerate(access_policies):
    axes[1].plot(sm_channel_loads, sm_avg_throughput[1, ap, :], linewidth=1.5, linestyle=linestyles[ap], color=colors[ap], label=access_policy)

axes[1].set_xlabel('channel load, $\kappa$')

axes[1].tick_params(axis='both', which='major', labelsize=8)
axes[1].tick_params(axis='both', which='minor', labelsize=8)

axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

axes[1].set_title('precoding-based ACK', fontsize=10)



# Go through all access policies
for ap, access_policy in enumerate(access_policies):
    axes[2].plot(sm_channel_loads, sm_avg_throughput[2, ap, :], linewidth=1.5, linestyle=linestyles[ap], color=colors[ap], label=access_policy)

axes[2].set_xlabel('channel load, $\kappa$')

axes[2].tick_params(axis='both', which='major', labelsize=8)
axes[2].tick_params(axis='both', which='minor', labelsize=8)

axes[2].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

axes[2].set_title('TDMA-based ACK', fontsize=10)

plt.subplots_adjust(
	left = 0.09,
	right = 0.99,
	bottom = 0.125,
	top = 0.925,
	wspace = 0.05,
	hspace = 0.05
)

plt.savefig('../figs/figure6_throughput.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure6_throughput.tex")

plt.show()
