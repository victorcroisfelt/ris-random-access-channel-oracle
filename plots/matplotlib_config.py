import matplotlib
import matplotlib.pyplot as plt

def plot_rc_config():
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,amsfonts,amsthm,bm,mathtools,cuted,bbold}')
    matplotlib.rc('font', **{'family': 'sans serif', 'serif': ['Times'], 'size': 10})

#  # LaTeX type definitions
# plt.rcParams.update({
#     'text.usetex': True,
#     'font.family': "serif",
#     'font.serif': ["Times"]
#     })
#
# #plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size': 14})
# #plt.rc('text', usetex=True)
# plt.rc('text.latex', preamble=r'\usepackage{amssymb,amsmath,amsfonts,amsthm,mathtools,cuted,bbold} \usepackage[cmintegrals]{newtxmath}')
