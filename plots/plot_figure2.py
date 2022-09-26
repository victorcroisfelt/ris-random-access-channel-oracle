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
data = np.load('../data/figure3.npz')

mask = data['mask']
tr_codebook = data['tr_codebook']
ac_codebook = data['ac_codebook']

true_snr_dl = data['true_snr_dl']
tr_snr_dl = data['tr_snr_dl']
ac_snr_dl = data['ac_snr_dl']

########################################
# Plot
########################################
ue_styles = ['-', '--']

ue_labels = [
    r'UE 0: $\theta_0=30^{\circ}$',
    r'UE 1: $\theta_1=60^{\circ}$'
    ]

fig, ax = plt.subplots(figsize=(6.1, 2.5))

for ue in range(2):

    ax.plot(np.rad2deg(mask), np.abs(true_snr_dl[ue])**2, linestyle=ue_styles[ue], linewidth=1.5, alpha=0.5, label=ue_labels[ue])

    if ue == 0:
        markerline, stemlines, baseline = ax.stem(np.rad2deg(tr_codebook), np.abs(tr_snr_dl[ue])**2, linefmt=':', markerfmt='o', label='training configs.')
    else:
        markerline, stemlines, baseline = ax.stem(np.rad2deg(tr_codebook), np.abs(tr_snr_dl[ue])**2, linefmt=':', markerfmt='o')

    baseline.set_color('none')

    stemlines.set_color('black')
    stemlines.set_alpha(0.25)

    markerline.set_markeredgecolor('grey')
    markerline.set_markerfacecolor('white')
    markerline.set_markersize(5)

    if ue == 0:
        markerline, stemlines, baseline = ax.stem(np.rad2deg(ac_codebook), np.abs(ac_snr_dl[ue])**2, linefmt=':', markerfmt='D', label='access configs.')
    else:
        markerline, stemlines, baseline = ax.stem(np.rad2deg(ac_codebook), np.abs(ac_snr_dl[ue])**2, linefmt=':', markerfmt='D')

    baseline.set_color('none')

    stemlines.set_color('black')
    stemlines.set_alpha(0.25)

    markerline.set_markeredgecolor('grey')
    markerline.set_markerfacecolor('None')
    markerline.set_markersize(7)

ax.set_xlabel(r'DL reflecting angle, $\theta^{\rm DL}_r$')
ax.set_ylabel('DL SNR')

ax.set_xticks(np.arange(0, 100, 10))
ax.xaxis.set_major_formatter(StrMethodFormatter(u"{x:.0f}°"))

#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)

ax.legend(framealpha=0.5)

ax.grid(color='#E9E9E9', linestyle=':', linewidth=1.0, alpha=0.5)

plt.subplots_adjust(
    left = 0.075,
    right = 0.99,
    bottom = 0.175,
    top = 0.99,
    wspace = 0.5,
    hspace = 0.05
    )

plt.savefig('../figs/figure3.pdf', dpi='figure', format='pdf', transparent='True')

tikzplotlib.save("../tikz/figure3.tex")
#tikzplotlib.save("../tikz/figure3.tex", flavor="context")

plt.show()
