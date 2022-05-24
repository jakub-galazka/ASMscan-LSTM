import matplotlib.pyplot as plt

from config import PLOT_STYLE


def config_plot(xlabel=None, ylabel=None, ylim=True, size_scale=1):
    plt.figure(figsize=((6.4 * size_scale), (4.8 * size_scale)))
    plt.style.use(PLOT_STYLE)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if ylim: plt.ylim([-.05, 1.05])

def save_plot(path):
    plt.legend()
    plt.savefig(path)
    plt.show()
