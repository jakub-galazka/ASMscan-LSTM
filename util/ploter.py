import matplotlib.pyplot as plt

from config import PLOT_STYLE


def config_plot(xlabel=None, ylabel=None, ylim=True, size_scale=1, turn_of_default_style=False):
    plt.figure(figsize=((6.4 * size_scale), (4.8 * size_scale)))
    if not turn_of_default_style: plt.style.use(PLOT_STYLE)
    if xlabel != None: plt.xlabel(xlabel)
    if ylabel != None: plt.ylabel(ylabel)
    if ylim: plt.ylim([-.05, 1.05])

def save_plot(path):
    plt.legend()
    plt.savefig(path)
    plt.show()
