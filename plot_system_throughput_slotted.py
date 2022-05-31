import numpy as np

import matplotlib.pyplot as plt
from matplotlib import rc

########################################
# Loading
########################################

# Slot duration
T = 1

# Config. duration
Tconfig = T

# Load data
aloha_blocked_data = np.load('data/aloha_blocked_contending_full.npz')
strongest_data = np.load('data/ris_strongest_N100_contending_full.npz')
rand_data = np.load('data/ris_rand_N100_contending_full.npz')

# Number of successful access attempts
total_num_successful_attempts_aloha_blocked = aloha_blocked_data['total_num_successful_attempts']
total_num_successful_attempts_strongest = strongest_data['total_num_successful_attempts']
total_num_successful_attempts_rand = rand_data['total_num_successful_attempts']

# Common parameters
num_configs_range = strongest_data['num_configs_range']
num_ues_range = strongest_data['num_ues_range']

# Prepare to store total access times
T_access_aloha = np.zeros(num_configs_range.size)
T_access_ris = np.zeros(num_configs_range.size)

# Go through all number of access frames
for aa, num_access_frame in enumerate(num_configs_range):
    T_access_aloha[aa] = (Tconfig + T) * num_access_frame
    T_access_ris[aa] = 2 * (Tconfig + T) * num_access_frame

# Prepare to save average throughput
throughput_aloha_blocked = np.zeros(total_num_successful_attempts_aloha_blocked.shape[0:2])
throughput_strongest = np.zeros(total_num_successful_attempts_strongest.shape[0:2])
throughput_rand = np.zeros(total_num_successful_attempts_rand.shape[0:2])

# Go through all number of access frames
for aa, num_access_frame in enumerate(num_configs_range):
    throughput_aloha_blocked[aa] = total_num_successful_attempts_aloha_blocked[aa].mean(axis=-1) / T_access_aloha[aa]
    throughput_strongest[aa] = total_num_successful_attempts_strongest[aa].mean(axis=-1) / T_access_ris[aa]
    throughput_rand[aa] = total_num_successful_attempts_rand[aa].mean(axis=-1) / T_access_ris[aa]

# Prepare to save optimal stuff
optz_throughput_aloha_blocked = np.max(throughput_aloha_blocked, axis=0)
optz_throughput_strongest = np.max(throughput_strongest, axis=0)
optz_throughput_rand = np.max(throughput_rand, axis=0)

optz_num_config_aloha_blocked = np.argmax(throughput_aloha_blocked, axis=0)
optz_num_config_strongest = np.argmax(throughput_strongest, axis=0)
optz_num_config_rand = np.argmax(throughput_rand, axis=0)

# Stack throughputs
throughput = np.stack((optz_throughput_strongest, optz_throughput_rand, optz_throughput_aloha_blocked), axis=0)
num_config = np.stack((optz_num_config_strongest, optz_num_config_rand, optz_num_config_aloha_blocked), axis=0)

########################################
# Plot
########################################

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 14})
rc('text', usetex=True)

# Open axes
fig, ax = plt.subplots(figsize=(3.15, 2.3))

colors = ['#7a5195', '#ef5675', 'black']
styles = ['-', '--', ':']
methods = ['SCP', 'CARP', 'URP']

# Go through all methods
for mm in range(3):
    ax.plot(num_ues_range, throughput[mm], linewidth=2, linestyle=styles[mm], color=colors[mm], label=methods[mm])

# Set axis
ax.set_xlabel(r'number of contending UEs, $K$')
ax.set_ylabel(r'avg. ${\mathrm{th}}(S^{\star})$ [pkt./slot]')

# Legend
ax.legend(fontsize='x-small', framealpha=0.5)

ax.set_xticks(np.arange(1, 11))

ax.set_yscale('log')

# Finally
plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)

plt.tight_layout()

plt.show(block=False)

#####

# fig, ax = plt.subplots(figsize=(3.15, 3.5/2))
#
# colors = ['#7a5195', '#ef5675', 'black']
# styles = ['-', '--', ':']
# methods = ['SCP', 'CARP', 'URP']
#
# # Go through all methods
# for mm in range(3):
#
#     ax.plot(num_ues_range, num_config[mm] + 1, linewidth=1.5, linestyle=styles[mm], color=colors[mm], label=methods[mm])
#
# # Set axis
# ax.set_xlabel(r'number of contending UEs, $K$')
# ax.set_ylabel(r'$s^{\star}$')
#
# # Legend
# ax.legend(fontsize='x-small', framealpha=0.5)
#
# # ax.set_xticks(np.arange(1, 11))
# # ax.set_yticks(np.arange(1, 11))
#
# #ax.set_yscale('log')
#
# # Finally
# plt.grid(color='#E9E9E9', linestyle='--', linewidth=0.5)
#
# plt.tight_layout()
#
# plt.show(block=False)