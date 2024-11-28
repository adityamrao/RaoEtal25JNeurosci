import numpy as np
import matplotlib.pyplot as plt


def polar_histogram(data, bins_number, area_proportional=True, label=None):
    """
    Creates a polar histogram.

    Args:
        data: 1D array-like, angular data in radians.
        bins_number: Integer, the number of bins for the histogram.
        area_proportional: Boolean, if True, makes bar areas proportional to frequency.
                            If False, makes bar radius proportional to frequency.
    """
    data %= (2 * np.pi)
    bins = np.linspace(0.0, 2 * np.pi, bins_number + 1)
    n, _ = np.histogram(data, bins)
    width = 2 * np.pi / bins_number

    ax = plt.subplot(1, 1, 1, projection='polar')

    if area_proportional:
        radii = np.sqrt(n / (np.pi * (width/2)**2))
        bars = ax.bar(bins[:bins_number], radii, width=width, bottom=0.0)
    else:
        bars = ax.bar(bins[:bins_number], n, width=width, bottom=0.0)

    for bar in bars:
        bar.set_alpha(0.5)
    if label:
        bars.set_label(label)
        plt.legend()
