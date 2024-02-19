"""
本文件是HDT工具包的数据可视化模块。
"""
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy as np


def plot_hyperspectral_curve(hyperspectral_data, wave=None, x_label=None, y_label=None, title=None):
    """
    该函数实现了可视化光谱曲线的功能。

    输入：
    - hyperspectral_data: 光谱矩阵表格。
    - wave: 光谱各波段波长，若为None则设为1到波段数的序列。
    - x_label: 横坐标标题。
    - y_label: 纵坐标标题。
    - title: 标题。

    """
    if not wave:
        wave = range(1, hyperspectral_data.shape[1] + 1)
    for i in range(hyperspectral_data.shape[0]):
        plt.plot(wave, hyperspectral_data[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.show()


def plot_quantile_quantile(target_data, threshold=0.1, scatter_color='#4b74b2', outlier_color='#ffdf92',
                           line_color='#db3124', x_label=None, y_label=None, title=None):
    """
    该函数实现了可视化目标检测含量QQ图的功能。

    参数：
    - target_data: 待检验的数据.
    - threshold: 距离阈值，用于标记散点颜色，默认为0.1。
    - scatter_color: 散点颜色，默认为'#4b74b2'。
    - outlier_color: 超过阈值的散点颜色，默认为'#ffdf92'。
    - line_color: 参考直线颜色，默认为'#db3124'。
    - x_label: 横坐标标题。
    - y_label: 纵坐标标题。
    - title: 标题。
    """

    quantiles = np.linspace(0, 1, len(target_data))
    sorted_data = np.sort(target_data)

    theoretical_quantiles = stats.norm.ppf(quantiles)[1:-1]
    sorted_data = sorted_data[1:-1]

    regression = LinearRegression()
    regression.fit(theoretical_quantiles.reshape(-1, 1), sorted_data.reshape(-1, 1))

    line_x = np.array([theoretical_quantiles.min(), theoretical_quantiles.max()])
    line_y = regression.predict(line_x.reshape(-1, 1))
    distances = np.abs(regression.predict(theoretical_quantiles.reshape(-1, 1)) - sorted_data.reshape(-1, 1))
    colors = np.where(distances > threshold, outlier_color, scatter_color)

    plt.scatter(theoretical_quantiles, sorted_data, alpha=0.5, c=colors.flatten())
    plt.plot(line_x, line_y, color=line_color, linestyle='--', label='Reference Line')
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.show()
