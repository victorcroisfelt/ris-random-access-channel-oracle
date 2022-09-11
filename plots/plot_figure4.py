import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from matplotlib.ticker import StrMethodFormatter

from matplotlib_config import plot_rc_config

import tikzplotlib

########################################
# Preamble
########################################
plot_rc_config()

########################################
# Load data
########################################
data = np.load('../data/figure4.npz')

ue_angles = data['ue_angles']
epsilon_range = data['epsilon_range']

appr_max_freq_1 = data['appr_max_freq_1']
appr_max_freq_2 = data['appr_max_freq_2']

########################################
# Plot
########################################
styles = ['--', '-.', ':']
labels = [
    r'$\tilde{F}^{10^{-1}}_{\max}$ in (23)',
    r'$\tilde{F}^{10^{-2}}_{\max}$ in (23)',
    r'$\tilde{F}^{10^{-3}}_{\max}$ in (23)']


fig, ax = plt.subplots(figsize=(6.5, 3))

ax.plot(np.rad2deg(ue_angles), appr_max_freq_1 * np.ones_like(ue_angles), linestyle='-', color='black', linewidth=1.5, label=r'$\tilde{F}_{\max}$ in (16)')

# Go through all epsilons
for ii, epsilon in enumerate(epsilon_range):
    ax.plot(np.rad2deg(ue_angles), appr_max_freq_2[ii, :], linestyle=styles[ii], color='black', linewidth=1.5, label=labels[ii])

#ax.set_ylabel('Approx. maximum \n frequency, ' + r'$\tilde{F}_{\max,k}$')

ax.set_ylabel(r'approx. maximum frequency, $\tilde{F}_{\max,k}$')
ax.set_xlabel(r"UE's angle, $\theta_k$ in degrees")

ax.set_xticks(np.arange(0, 100, 10))
ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))

# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

ax.set_yscale('log')

ax.legend(framealpha=0.5)

ax.grid(color='#E9E9E9', linestyle=':', linewidth=1.0, alpha=0.5)

#plt.tight_layout()

plt.subplots_adjust(
    left = 0.125,
    right = 0.99,
    bottom = 0.175,
    top = 0.95,
    wspace = 0.5,
    hspace = 0.05
    )

#plt.savefig('max_freq.pdf', dpi='figure', format='pdf', transparent='True')

plt.show()

#tikzplotlib.save("figure3.tex")
#tikzplotlib.save("figure3.tex", flavor="context")
