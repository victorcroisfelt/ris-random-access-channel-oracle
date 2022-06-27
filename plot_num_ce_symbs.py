# -*- coding: utf-8 -*-
"""shannon-nyquist.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1rrWQ-d-wkfkvCnTX6grWVn6lpai5unHP
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amsmath} \usepackage{amssymb}')
matplotlib.rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 8})

# matplotlib.rc('xtick', labelsize=8)
# matplotlib.rc('ytick', labelsize=8)
#
# matplotlib.rc('text', usetex=True)
#
matplotlib.rcParams['text.latex.preamble'] = [
    r'\usepackage{amsfonts}',
    r'\usepackage{amsmath}',
    r'\usepackage{amssymb}'
]

########################################
# Parameters
########################################

# Define SNR range
snr_db = np.linspace(-10, 10)

# Transform SNR linear
snr = 10**(snr_db/10)

# Define tolerance values
tolerance_range = np.array([0.1, 0.01, 0.001])

# Compute lower bound
lower_bound = np.zeros((tolerance_range.size, snr.size))

# Go through all tolerances
for ii, tolerance in enumerate(tolerance_range):

    # Compute bound
    lower_bound[ii, :] = np.ceil(1 / (snr * tolerance))

########################################
# PLot signal
########################################
fig, ax = plt.subplots(figsize=(3.15, 3))

styles = ['-', '--', ':']
labels = ['10^{-1}', '10^{-2}', '10^{-3}']

# Go through all tolerances
for ii, tolerance in enumerate(tolerance_range):
    ax.plot(snr_db, lower_bound[ii, :], linestyle=styles[ii], color='black', linewidth=1.5, label=r'$\mathrm{tol}=' + str(labels[ii]) + '$')

ax.set_xlabel(r'$\mathrm{SNR}^{\mathrm{DL}}$ [dB]')
ax.set_ylabel(r'$L_{\mathrm{ce}}$ lower bound')

# ax.set_xscale('log')
ax.set_yscale('log')

ax.legend(fontsize='x-small', framealpha=0.5)

ax.grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()
