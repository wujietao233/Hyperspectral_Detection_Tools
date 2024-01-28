"""
本文件是HDT工具包的数据加载模块，包括原始光谱数据的批量ROI区域提取，原始光谱矩阵的保存和加载，以及数据清洗的功能。
"""
import numpy as np
import pandas as pd

from preprocess import largest_connected_component_cropping
from spectral.io import envi
import os


def region_of_interest_extraction(
        rootPath: str, targetPath: str, threshold: float = -1.0, scaling: float = 1.0,
        band: int = -1):
    """
    该函数实现了利用二值化和阈值分割实现感兴趣区域提取的功能。

    请确保rootPath文件夹下只有hdr文件和raw文件，且hdr文件和raw文件的文件名一一对应。
    请确保rootPath文件夹下的每个hdr/raw文件的文件名都和target.csv的第一列的编号一一对应。

    - rootPath: 存放hdr文件和raw文件的文件夹。
    - targetPath: 存放所预测目标的csv文件，第一列为编号，第二列为待检测的目标。
    - threshold: 二值化分割阈值，-1表示根据通道像素的中位数自动计算阈值，默认值为-1.0。
    - scaling: 缩放系数，即分割阈值和像素通道中位数的比例，默认值为1.0。
    - band: 二值化通道索引，-1表示通道的中位数，默认值为-1。
    """

    assert os.path.exists(rootPath), f"{rootPath} is not exist!\n"
    assert os.path.exists(targetPath), f"{targetPath} is not exist!\n"
    # 读取targetPath文件
    target = pd.read_csv(targetPath)
    # target转为字典
    targetDict = dict(zip(target.iloc[:, 0], target.iloc[:, 1]))
    # 表格列表
    tableList = []
    # 获取rootPath路径下的所有hdr文件
    hdrList = list(filter(lambda x: x.endswith('.hdr'), os.listdir(rootPath)))
    # 计算columns
    columns = []
    # 遍历每一个hdr文件
    for hdr in hdrList:
        # 计算hdr文件的路径
        hdrPath = os.path.join(rootPath, hdr)
        # 计算raw文件的路径
        rawPath = hdrPath.replace('.hdr', '.raw')
        # 读取高光谱图片
        img = envi.open(hdrPath, rawPath)
        # 获取通道
        columns = img.bands.centers
        # 计算二值化波段
        channel_band = band
        if band == -1:
            channel_band = img.nbands // 2
        # 读取单通道图
        channel = img[:, :, channel_band]
        # 获取阈值
        channel_threshold = threshold
        if threshold == -1.0:
            channel_threshold = scaling * channel.mean()
        # 二值化掩码图
        mask = np.where(np.squeeze(channel) > channel_threshold, 1, 0)
        # 获取裁剪后数据
        cropped_images = largest_connected_component_cropping(mask, img)
        # 获取平均反射率
        average_reflectance = cropped_images.mean(axis=0).mean(axis=0).tolist()
        # 获取编号
        ID_number = hdr.split('.hdr')[0]
        # 列表添加元素
        tableList.append([ID_number] + average_reflectance + [
            targetDict[ID_number] if ID_number in targetDict else None
        ])
    # 列表转为表格并添加表头
    table = pd.DataFrame(tableList, columns=[target.columns[0]] + columns + [target.columns[1]])
    # 返回table
    return table


