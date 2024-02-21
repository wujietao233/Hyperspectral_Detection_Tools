from . import feature_dimension_reduction
from . import load
from . import plot
from . import preprocess
from . import split
from . import train
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 200
plt.rcParams['font.sans-serif'] = ['times new roman']

__all__ = [
    "feature_dimension_reduction",
    "load",
    "plot",
    "preprocess",
    "split",
    "train"
]
