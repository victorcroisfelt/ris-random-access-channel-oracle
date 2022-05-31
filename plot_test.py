import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc

import numpy as np

# LaTeX type definitions
rc('font', **{'family': 'sans serif', 'serif': ['Computer Modern'], 'size': 12})
rc('text', usetex=True)

# Load stuff
ff_data = np.load('FF.npz', allow_pickle=True)
cnn_data = np.load('CNN.npz', allow_pickle=True)
rnn_data = np.load('RNN.npz', allow_pickle=True)

# Get data
ff_sgd = ff_data['sgd']
cnn_sgd = cnn_data['sgd']
rnn_sgd = rnn_data['sgd']

ff_adam = ff_data['adam']
cnn_adam = cnn_data['adam']
rnn_adam = rnn_data['adam']
breakpoint()

fig, axes = plt.subplots(ncols=2)

styles = ['-', '--', ':']

axes[0].plot(range(len(ff_sgd)), 100*ff_sgd, linestyle=styles[0], label='FNN')
axes[0].plot(range(len(ff_sgd)), 100*cnn_sgd, linestyle=styles[1], label='CNN')
axes[0].plot(range(len(ff_sgd)), 100*rnn_sgd, linestyle=styles[2], label='RNN')

plt.gca().set_prop_cycle(None)

axes[1].plot(range(len(ff_sgd)), 100*ff_adam, linestyle=styles[0])
axes[1].plot(range(len(ff_sgd)), 100*cnn_adam, linestyle=styles[1])
axes[1].plot(range(len(ff_sgd)), 100*rnn_adam, linestyle=styles[2])

axes[0].set_xlabel('epochs')
axes[1].set_xlabel('epochs')

axes[0].set_ylabel('test accuracy (\%)')
axes[1].set_ylabel('test accuracy (\%)')

axes[0].set_xticks([0, 10, 20, 30, 40, 50])
axes[1].set_xticks([0, 10, 20, 30, 40, 50])

axes[0].legend(framealpha=0.5)

axes[0].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
axes[1].grid(color='gray', linestyle=':', linewidth=0.5, alpha=0.5)

plt.tight_layout()

plt.show()