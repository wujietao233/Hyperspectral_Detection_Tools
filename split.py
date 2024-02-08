"""
本文件是HDT工具包的数据划分模块。
"""
from scipy.spatial.distance import cdist
import numpy as np


def sample_partitioning_based_on_joint_x_y_distance(X, y, test_size=0.25, metric='euclidean', *args, **kwargs):
    """
    该函数实现了基于SPXY算法（sample set partitioning based on joint x-y distance）的样本划分功能。

    输入：
    - X: i行j列的数组，表示i个光谱样本和j个变量（波长/波数/拉曼位移等）。
    - y: 形状为i行的数组，表示i个目标变量值。
    - test_size: 如果为浮点数，则选择(i * (1-test_size))个样本作为测试数据，默认为0.25。如果为整数，则直接使用test_size作为测试数据的样本数量。
    - metric: 距离度量指标，默认为'euclidean'。

    输出：
    - train_index: 作为训练数据的样本的索引列表（从0开始计数）。
    - test_index: 作为测试数据的样本的索引列表（从0开始计数）。
    """

    def max_min_distance_split(distance, train_size):
        """
        该函数实现了基于最大最小距离的样本集划分的功能，是Kennard Stone方法的核心。

        输入：
        - distance: 半正定实对称矩阵，表示一定距离度量的距离矩阵。
        - train_size: 训练数据的样本数量，应大于2。

        输出：
        - train_index: 作为训练数据的样本的索引列表（从0开始计数）。
        - test_index: 作为测试数据的样本的索引列表（从0开始计数）。
        """

        train_index = []
        test_index = [x for x in range(distance.shape[0])]

        first_2pts = np.unravel_index(np.argmax(distance), distance.shape)
        train_index.append(first_2pts[0])
        train_index.append(first_2pts[1])

        test_index.remove(first_2pts[0])
        test_index.remove(first_2pts[1])

        for i in range(train_size - 2):
            select_distance = distance[train_index, :]
            min_distance = select_distance[:, test_index]
            min_distance = np.min(min_distance, axis=0)
            max_min_distance = np.max(min_distance)

            points = np.argwhere(select_distance == max_min_distance)[:, 1].tolist()
            for point in points:
                if point in train_index:
                    pass
                else:
                    train_index.append(point)
                    test_index.remove(point)
                    break
        return train_index, test_index

    if test_size < 1:
        train_size = round(X.shape[0] * (1 - test_size))
    else:
        train_size = X.shape[0] - round(test_size)
    if train_size > 2:
        y = y.reshape(y.shape[0], -1)
        distance_X = cdist(X, X, metric=metric, *args, **kwargs)
        distance_y = cdist(y, y, metric=metric, *args, **kwargs)
        distance_X = distance_X / distance_X.max()
        distance_y = distance_y / distance_y.max()

        distance = distance_X + distance_y
        train_index, test_index = max_min_distance_split(distance, train_size)
    else:
        raise ValueError("train sample size should be at least 2")
    return train_index, test_index
