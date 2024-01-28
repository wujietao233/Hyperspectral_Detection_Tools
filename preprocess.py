"""
本文件是HDT工具包的数据预处理模块，包括ROI区域提取、光谱曲线预处理等功能。
"""
import numpy as np
import cv2


def largest_connected_component_cropping(mask, img):
    """
    最大连通域裁剪

    - mask: 二值化图像。
    - img: 高光谱图像数据。

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
