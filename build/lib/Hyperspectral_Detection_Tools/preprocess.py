"""
本文件是HDT工具包的数据预处理模块，包括ROI区域提取、光谱曲线预处理等功能。
"""
import numpy as np
import cv2
import pywt
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter


def largest_connected_component_cropping(mask, img):
    """
    该函数实现了ROI区域提取中，最大连通域裁剪的功能。

    输入：
    - mask: 二值化图像。
    - img: 高光谱图像数据。

    输出：
    - cropped_images: 裁剪后图像数据。

    """
    # 获取连通域列表
    contours, hierarchy = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    # 获取最大连通域
    largest_contours = sorted(contours, key=len, reverse=True)[0].squeeze()
    # 获取最大外接矩形
    x1 = largest_contours[:, 0].min()
    y1 = largest_contours[:, 1].min()
    x2 = largest_contours[:, 0].max()
    y2 = largest_contours[:, 1].max()
    # 获取掩码
    cropped_images = np.empty([y2 - y1, x2 - x1, img.shape[2]])
    # 裁剪数据
    for i in range(img.shape[2]):
        cropped_images[:, :, i:i + 1] = img[y1:y2, x1:x2, i:i + 1]
    return cropped_images


def preprocess_hyperspectral_table(origin_hyperspectral_data,
                                   preprocess_method: str,
                                   window_length: int = 21,
                                   polyorder: int = 3,
                                   threshold: float = 0.04):
    """
    该函数实现了光谱矩阵表格预处理的功能。

    已实现的预处理方法包括：
    最小值最大值归一化、标准化、SG平滑、滑动平均平滑、多元散射校正、趋势矫正、标准正态变换、小波变换、一阶导数和二阶导数。

    输入：
    - origin_hyperspectral_data: 光谱矩阵表格。
    - preprocess_method: 预处理方法，可以为：MM、SS、SG、MA、MSC、DT、SNV、WT、FD和SD，也可以使用+将多种预处理方法连接起来，如：SS+SNV。
    - window_length: SG平滑/滑动平均平滑窗口大小。
    - polyorder: SG平滑多项式拟合的阶数。
    - threshold: 小波变换阈值。

    输出：
    - preprocess_hyperspectral_data: 预处理后光谱矩阵表格。

    """

    def preprocess_min_max_scaler(origin_hyperspectral_data):
        """
        该函数实现了光谱矩阵表格最小值最大值归一化预处理的功能。
        """
        return preprocessing.MinMaxScaler().fit_transform(origin_hyperspectral_data)

    def preprocess_standard_scaler(origin_hyperspectral_data):
        """
        该函数实现了光谱矩阵表格标准化预处理的功能。
        """
        return preprocessing.scale(origin_hyperspectral_data)

    def preprocess_savitzky_golay_smooth(origin_hyperspectral_data, window_length: int, polyorder: int):
        """
        该函数实现了光谱矩阵表格SG平滑的功能。
        """
        return savgol_filter(origin_hyperspectral_data, window_length=window_length, polyorder=polyorder)

    def preprocess_moving_average_smooth(origin_hyperspectral_data, window_length: int):
        """
        该函数实现了光谱矩阵表格滑动平均平滑的功能。
        """
        kernel = np.ones(window_length) / window_length
        return np.apply_along_axis(lambda x: np.convolve(x, kernel, mode='valid'), axis=1,
                                   arr=origin_hyperspectral_data)

    def preprocess_multiplicative_scatter_correction(origin_hyperspectral_data):
        """
        该函数实现了光谱矩阵表格多元散射校正的功能。
        """
        x = np.mean(origin_hyperspectral_data, axis=0).reshape(-1, 1)
        return np.vstack([
            (y - lin.intercept_) / lin.coef_ for lin, y in zip(
                [LinearRegression().fit(x, y) for y in origin_hyperspectral_data],
                origin_hyperspectral_data)
        ])

    def preprocess_detrend_correction(origin_hyperspectral_data):
        """
        该函数实现了光谱矩阵表格趋势校正的功能。
        """
        x = np.arange(origin_hyperspectral_data.shape[1], dtype=np.float32)
        return np.vstack([
            y - (x * lin.coef_[0] + lin.intercept_) for lin, y in zip(
                [LinearRegression().fit(x.reshape(-1, 1), y) for y in origin_hyperspectral_data],
                origin_hyperspectral_data)
        ])

    def preprocess_standard_normal_variate_transform(origin_hyperspectral_data):
        """
        该函数实现了光谱矩阵表格趋势校正的功能。
        """
        return (origin_hyperspectral_data - np.mean(origin_hyperspectral_data, axis=1, keepdims=True)) / np.std(
            origin_hyperspectral_data, axis=1, keepdims=True)

    def preprocess_wavelet_transform(origin_hyperspectral_data, threshold: float):
        """
        该函数实现了光谱矩阵表格小波变换的功能。
        """
        return np.vstack([
            pywt.waverec(c, 'db8') for c in [
                [c[0]] + [pywt.threshold(c[i], threshold * max(c[i])) for i in range(1, len(c))]
                for c in [
                    pywt.wavedec(
                        sample,
                        wavelet='db8',
                        level=pywt.dwt_max_level(
                            origin_hyperspectral_data.shape[1],
                            pywt.Wavelet('db8').dec_len
                        )
                    ) for sample in origin_hyperspectral_data
                ]
            ]
        ])

    def preprocess_first_derivative(origin_hyperspectral_data):
        """
        该函数实现了光谱矩阵表格一阶求导的功能。
        """
        return np.diff(origin_hyperspectral_data, axis=1)

    def preprocess_second_derivative(origin_hyperspectral_data):
        """
        该函数实现了光谱矩阵表格二阶求导的功能。
        """
        return np.diff(np.diff(origin_hyperspectral_data, axis=1), axis=1)

    if "+" in preprocess_method:
        for mtd in preprocess_method.split("+"):
            origin_hyperspectral_data = preprocess_hyperspectral_table(origin_hyperspectral_data, mtd)
        return origin_hyperspectral_data
    preprocessDict = {
        'MM': preprocess_min_max_scaler(origin_hyperspectral_data),
        'SS': preprocess_standard_scaler(origin_hyperspectral_data),
        'SG': preprocess_savitzky_golay_smooth(origin_hyperspectral_data, window_length, polyorder),
        'MA': preprocess_moving_average_smooth(origin_hyperspectral_data, window_length),
        'MSC': preprocess_multiplicative_scatter_correction(origin_hyperspectral_data),
        'DT': preprocess_detrend_correction(origin_hyperspectral_data),
        'SNV': preprocess_standard_normal_variate_transform(origin_hyperspectral_data),
        'WT': preprocess_wavelet_transform(origin_hyperspectral_data, threshold),
        'FD': preprocess_first_derivative(origin_hyperspectral_data),
        'SD': preprocess_second_derivative(origin_hyperspectral_data),
    }
    preprocess_hyperspectral_data = preprocessDict[preprocess_method.upper()]
    return preprocess_hyperspectral_data
